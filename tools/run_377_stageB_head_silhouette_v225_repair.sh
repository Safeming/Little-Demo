#!/usr/bin/env bash
set -u
set -o pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-stageB_headsil_v225_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SEED="${SEED:--1}"
GPUS="${GPUS:-0,1,2,3}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-2500}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-40}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
QUEUE_LAUNCH_STAGGER_SECONDS="${QUEUE_LAUNCH_STAGGER_SECONDS:-30}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
SMOKE="${SMOKE:-0}"
DO_RENDER="${DO_RENDER:-1}"
RUN_DIAGNOSTIC_TESTVIEW="${RUN_DIAGNOSTIC_TESTVIEW:-1}"

export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS_PER_JOB"
export BLIS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENCV_FOR_THREADS_NUM="$CPU_THREADS_PER_JOB"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
V223A_EXP="${V223A_EXP:-$ROOT/exp/stageB/377_hulk_light_v223a_semantic_all20_long_stageB_overnight_final_20260511_003203_bjt}"
V223A_RENDER="${V223A_RENDER:-${V223A_EXP}_render_full_best}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v224c_head_reliable_views_preserve_stageB_headfix_fixed_20260511_103654_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/best_ckpt.pth}"
BASE_RENDER="${BASE_RENDER:-${BASE_EXP}_render_parserhard_best}"
BASE_ITER="${BASE_ITER:-111710}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_head_silhouette_v225_repair_$RUN_ID}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
PIDS="$LOG_DIR/pids.tsv"
STATUS_JSON="$LOG_DIR/status.json"
mkdir -p "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
RELIABLE_PLUS="[2,3,6,8,11,13,18,19,20]"
HEAD_DIAG="[18,19,20,21,22,23]"
HEAD_SELECT=(render_c21_f000240.png render_c21_f000300.png render_c22_f000240.png render_c23_f000300.png render_c23_f000420.png)
BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin]"
START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"

printf 'time_bjt\tgpu\tname\tphase\tdetail\n' > "$EVENTS"
printf 'name\tgpu\tpid\n' > "$PIDS"
printf 'name\tkind\texp_dir\trender_exp\tstatus\tdetail\n' > "$SUMMARY"
printf 'RUN_ID=%s\nSTART_BJT=%s\nGPUS=%s\nBASE_EXP=%s\nBASE_CKPT=%s\nBASE_RENDER=%s\nBASE_ITER=%s\nDATA_ROOT=%s\nPARSER_ROOT=%s\nCOMPACT_MAPPING=%s\nSMOKE=%s\nDO_RENDER=%s\n' \
  "$RUN_ID" "$START_BJT" "$GPUS" "$BASE_EXP" "$BASE_CKPT" "$BASE_RENDER" "$BASE_ITER" \
  "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$SMOKE" "$DO_RENDER" | tee "$LOG_DIR/run_info.txt"

log_event() {
  local gpu="$1"
  local name="$2"
  local phase="$3"
  local detail="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$gpu" "$name" "$phase" "$detail" | tee -a "$EVENTS"
}

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$1" "$2" "$3" <<'PY'
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$SUMMARY"
}

split_csv() {
  local value="$1"
  local IFS=','
  read -ra _items <<< "$value"
  printf '%s\n' "${_items[@]}"
}

gpu_stats() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
}

wait_for_gpu() {
  local gpu="$1"
  local name="$2"
  if [ "$WAIT_FOR_FREE_GPUS" != "1" ] || [ "$SMOKE" = "1" ]; then
    return 0
  fi
  local used util
  while true; do
    read -r used util < <(gpu_stats "$gpu")
    used="${used:-999999}"
    util="${util:-100}"
    if [ "$used" -le "$GPU_MAX_USED_MB_START" ] && [ "$util" -le "$GPU_MAX_UTIL_START" ]; then
      log_event "$gpu" "$name" "gpu_ready" "used=${used}MiB util=${util}%"
      return 0
    fi
    log_event "$gpu" "$name" "gpu_wait" "used=${used}MiB util=${util}% threshold=${GPU_MAX_USED_MB_START}MiB/${GPU_MAX_UTIL_START}%"
    sleep "$GPU_WAIT_POLL_SECONDS"
  done
}

run_head_diag() {
  local name="$1"
  local gpu="$2"
  local render_exp="$3"
  local label="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/analyze_377_head_silhouette_artifact.py \
    --render-exp "$render_exp" \
    --baseline-render-exp "$V223A_RENDER" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --select "${HEAD_SELECT[@]}" \
    --out-dir "$render_exp/diagnostics/head_silhouette_${label}" \
    --panels gt gt_mask render outside_gt base_render base_diff layer region compact_semantic semantic \
    > "$LOG_DIR/${name}_head_diag_${label}.log" 2>&1 || true
}

run_contour_diag() {
  local name="$1"
  local gpu="$2"
  local render_exp="$3"
  local label="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --band-width 7 \
    --topk 12 > "$LOG_DIR/${name}_contour_${label}.log" 2>&1 || true
}

render_export() {
  local name="$1"
  local gpu="$2"
  local config_exp="$3"
  local ckpt="$4"
  local out_exp="$5"
  local label="$6"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_${label}"

  if [ ! -f "$ckpt" ]; then
    summary_row "$name" "$label" "$config_exp" "$out_exp" "render_skipped" "missing_ckpt=$ckpt"
    log_event "$gpu" "$name" "render_skipped" "missing_ckpt=$ckpt"
    return 0
  fi

  wait_for_gpu "$gpu" "${name}_render_${label}"
  log_event "$gpu" "$name" "render_start" "$out_exp"
  write_status "$gpu" "render_start" "$name:$label"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" render.py \
    --config-path "$config_exp/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$out_exp" \
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
    "++binding_map_hard_fg_opacity_threshold=0.105" \
    "++binding_map_opacity_threshold=0.105" \
    "++binding_map_compact_semantic_opacity_threshold=0.105" \
    "++binding_map_hard_fg_close_kernel=1" \
    "++binding_map_hard_fg_erode_kernel=3" \
    "++binding_map_support_close_kernel=1" \
    "++binding_map_support_erode_kernel=3" \
    "++binding_map_mask_source=parser_hard" \
    "++binding_map_mask_erode_kernel=3" \
    "++render_export_opacity_threshold=0.080" \
    "++render_export_close_kernel=1" \
    "++render_export_erode_kernel=1" \
    "export_semantic_editable_assets=true" \
    "semantic_editable_parser_root=$PARSER_ROOT" \
    "semantic_editable_parser_layout=cihp_subject" \
    "semantic_editable_direct_parser_mode=true" \
    "semantic_editable_export_compact_head=true" \
    "semantic_editable_include_binding_summary=true" \
    "+semantic_editable_preview_min_area=18" \
    "hydra.run.dir=$hydra_run_dir" \
    wandb_disable=true > "$LOG_DIR/${name}_render_${label}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    summary_row "$name" "$label" "$config_exp" "$out_exp" "render_failed" "status=$status"
    log_event "$gpu" "$name" "render_failed" "status=$status"
    return "$status"
  fi

  "$PYTHON_BIN" tools/make_binding_paper_montage.py \
    --exp-dir "$out_exp" \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --split test-view \
    --panels gt render layer region body_prob cloth_prob compact_semantic thin semantic \
    --select "${HEAD_SELECT[@]}" \
    --output-dir "$out_exp/test-view/paper_montages_selected" > "$LOG_DIR/${name}_montage_${label}.log" 2>&1 || true
  run_contour_diag "$name" "$gpu" "$out_exp" "$label"
  run_head_diag "$name" "$gpu" "$out_exp" "$label"
  summary_row "$name" "$label" "$config_exp" "$out_exp" "ok" "$ckpt"
  log_event "$gpu" "$name" "render_done" "$out_exp"
}

train_repair() {
  local name="$1"
  local gpu="$2"
  local train_views="$3"
  local iterations="$4"
  local checkpoint_list="$5"
  shift 5
  local overrides=("$@")
  if [ "$SMOKE" = "1" ]; then
    iterations=4
    checkpoint_list="[2,4]"
  fi

  local exp_dir="$ROOT/exp/stageB/377_hulk_light_${name}_${RUN_ID}"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_train"
  local final_ckpt="$exp_dir/ckpt$((BASE_ITER + iterations)).pth"
  mkdir -p "$exp_dir"

  wait_for_gpu "$gpu" "$name"
  log_event "$gpu" "$name" "train_start" "iterations=$iterations views=$train_views"
  write_status "$gpu" "train_start" "$name"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" train.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$train_views" \
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
    "dataset.parsing_prior.skip_empty_min_pixels=96" \
    "start_checkpoint=$BASE_CKPT" \
    "exp_dir=$exp_dir" \
    "seed=$SEED" \
    "wandb_disable=true" \
    "hydra.run.dir=$hydra_run_dir" \
    "++resume.allow_partial_converter_load=true" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.,pose_correction.pose_body_train_mask]" \
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
    "opt.iterations=$iterations" \
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
    "++opt.semantic_region_logits_lr=0.00022" \
    "++opt.semantic_compact_logits_lr=0.00022" \
    "++opt.lambda_binding_semantic_adapter_reg=0.0025" \
    "++opt.stageB_semantic_loss_enable=true" \
    "++opt.stageB_semantic_ignore_uncertain=true" \
    "++opt.stageB_semantic_ignore_boundary_width=11" \
    "++opt.stageB_semantic_use_opacity_support=true" \
    "++opt.stageB_semantic_opacity_threshold=0.045" \
    "++opt.stageB_semantic_min_valid_pixels=96" \
    "++opt.stageB_semantic_body_cloth_weight=0.34" \
    "++opt.stageB_semantic_compact_weight=0.42" \
    "++opt.stageB_semantic_parent_consistency_weight=0.24" \
    "++opt.stageB_semantic_exclusive_weight=0.08" \
    "++opt.stageB_semantic_adapter_smooth_weight=0.026" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_min_prob=0.020" \
    "++opt.train_sample_camera_max_prob=0.160" \
    "++opt.train_sample_log_interval=100" \
    "++opt.train_sample_accumulation_steps=4" \
    "++opt.foreground_mask_source=hard" \
    "++opt.global_mask_source=hard" \
    "++opt.boundary_target_mask_source=hard" \
    "++opt.face_region_source=union" \
    "++opt.face_region_parser_dilate=3" \
    "++opt.boundary_region_source=binary" \
    "++opt.boundary_band_width=11" \
    "opt.lambda_mask=0.008" \
    "++opt.mask_loss_type=l1" \
    "++opt.lambda_mask_boundary=0.016" \
    "++opt.lambda_mask_boundary_hard=0.010" \
    "++opt.lambda_silhouette_outer=0.016" \
    "++opt.lambda_silhouette_outer_shell=0.034" \
    "++opt.lambda_silhouette_head_outer_shell=0.090" \
    "++opt.lambda_silhouette_outer_spike=0.024" \
    "++opt.silhouette_outer_ring_width=9" \
    "++opt.silhouette_outer_shell_start_width=1" \
    "++opt.silhouette_outer_shell_end_width=27" \
    "++opt.silhouette_outer_shell_soft_weights=false" \
    "++opt.silhouette_head_outer_region_dilate=29" \
    "++opt.silhouette_head_outer_bottom_ratio=0.42" \
    "++opt.silhouette_head_outer_fg_clip_dilate=55" \
    "++opt.silhouette_head_outer_min_pixels=16" \
    "++opt.boundary_image_error_score_enable=true" \
    "++opt.boundary_image_error_score_signed_enable=true" \
    "++opt.boundary_image_error_score_mix=1.0" \
    "++opt.boundary_image_error_score_gain=1.24" \
    "++opt.boundary_image_error_score_power=1.04" \
    "++opt.boundary_image_error_score_min=0.020" \
    "++opt.boundary_image_error_score_band_width=7" \
    "++opt.boundary_image_error_pred_threshold=0.36" \
    "++opt.boundary_image_error_target_threshold=0.50" \
    "++opt.boundary_image_error_score_focus_dilate=3" \
    "++opt.boundary_image_error_score_smooth_k=8" \
    "++opt.boundary_image_error_score_smooth_blend=0.16" \
    "++opt.boundary_image_error_score_prior_floor=0.04" \
    "++opt.boundary_aware_enable=true" \
    "++opt.boundary_aware_gate_l1_boundary=true" \
    "++opt.boundary_aware_gate_mask_boundary=true" \
    "++opt.boundary_aware_gate_mask_boundary_hard=true" \
    "++opt.boundary_aware_gate_silhouette_outer=true" \
    "++opt.boundary_aware_gate_silhouette_outer_shell=true" \
    "++opt.boundary_aware_gate_silhouette_head_outer_shell=true" \
    "++opt.boundary_aware_gate_silhouette_outer_spike=true" \
    "++opt.boundary_aware_threshold=0.018" \
    "++opt.boundary_aware_score_power=0.90" \
    "++opt.boundary_tag_schedule_use_local_iteration=true" \
    "++opt.boundary_tag_enable=true" \
    "++opt.boundary_tag_init_iter=1" \
    "++opt.boundary_tag_update_interval=20" \
    "++opt.boundary_tag_update_until_iter=1400" \
    "++opt.boundary_tag_mode=topk_ratio" \
    "++opt.boundary_tag_topk_ratio=0.12" \
    "++opt.boundary_tag_min_ratio=0.08" \
    "++opt.boundary_tag_threshold=0.20" \
    "++opt.boundary_tag_binary=true" \
    "++opt.boundary_tag_use_score_within_subset=true" \
    "++opt.boundary_tag_score_smooth_blend=0.16" \
    "++opt.boundary_tag_score_smooth_k=12" \
    "++opt.boundary_tag_support_k=8" \
    "++opt.boundary_tag_support_threshold=0.18" \
    "++opt.boundary_signed_routing_enable=true" \
    "++opt.boundary_signed_mixed_loss_scale=0.035" \
    "++opt.boundary_signed_shrink_loss_scale=1.65" \
    "++opt.boundary_signed_grow_loss_scale=0.16" \
    "++opt.boundary_signed_share_gain=1.04" \
    "++opt.boundary_signed_share_power=1.08" \
    "++opt.boundary_signed_shrink_share_gain=1.34" \
    "++opt.boundary_signed_shrink_share_power=1.42" \
    "++opt.boundary_aware_freeze_converter_for_boundary_loss=true" \
    "++opt.boundary_aware_feature_dc_scale=0.0" \
    "++opt.boundary_aware_feature_rest_scale=0.0" \
    "++opt.boundary_aware_xyz_scale=0.0" \
    "++opt.boundary_aware_opacity_scale=0.22" \
    "++opt.boundary_aware_scaling_scale=0.08" \
    "++opt.boundary_aware_boundary_opacity_residual_scale=0.90" \
    "++opt.boundary_aware_boundary_scaling_residual_scale=0.48" \
    "opt.opacity_lr=0.00012" \
    "opt.scaling_lr=0.000006" \
    "++opt.boundary_opacity_residual_lr=0.000045" \
    "++opt.boundary_scaling_residual_lr=0.000010" \
    "++opt.lambda_boundary_opacity_residual_reg=0.0012" \
    "++opt.lambda_boundary_scaling_residual_reg=0.00045" \
    "++opt.lambda_boundary_opacity_residual_smooth=0.0010" \
    "++opt.lambda_boundary_scaling_residual_smooth=0.0007" \
    "opt.lambda_l1=0.0" \
    "opt.lambda_l1_fg=0.0" \
    "opt.lambda_l1_boundary=0.0" \
    "opt.lambda_l1_face=0.0" \
    "opt.lambda_perceptual=0.0" \
    "opt.lambda_edge_face=0.0" \
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
    "test_interval=300" \
    "test_iterations=$checkpoint_list" \
    "save_iterations=$checkpoint_list" \
    "checkpoint_iterations=$checkpoint_list" \
    "++validation_image_log_limit=0" \
    "opt.grad_clip=0.0035" \
    "${overrides[@]}" > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    summary_row "$name" "train" "$exp_dir" "" "train_failed" "status=$status"
    log_event "$gpu" "$name" "train_failed" "status=$status"
    return "$status"
  fi
  summary_row "$name" "train" "$exp_dir" "" "ok" "best=$exp_dir/best_ckpt.pth final=$final_ckpt"
  log_event "$gpu" "$name" "train_done" "$LOG_DIR/${name}.log"
  if [ "$DO_RENDER" = "1" ] && [ "$SMOKE" != "1" ]; then
    render_export "$name" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "${exp_dir}_render_parserhard_best" "best"
    render_export "$name" "$gpu" "$exp_dir" "$final_ckpt" "${exp_dir}_render_parserhard_final" "final"
  fi
}

run_named_job() {
  local gpu="$1"
  local job="$2"
  case "$job" in
    v225a_head_mask_opacity_moderate)
      train_repair "$job" "$gpu" "$RELIABLE_PLUS" 900 "[300,600,900]" \
        "++opt.train_sample_camera_weights={2:0.85,3:0.95,6:0.90,8:1.00,11:1.05,13:0.80,18:1.20,19:1.35,20:1.45}" \
        "++opt.lambda_silhouette_head_outer_shell=0.075" \
        "++opt.boundary_aware_opacity_scale=0.18" \
        "++opt.boundary_aware_scaling_scale=0.06" \
        "opt.opacity_lr=0.000075" \
        "opt.scaling_lr=0.000004" \
        "++opt.boundary_opacity_residual_lr=0.000032" \
        "opt.grad_clip=0.0028"
      ;;
    v225b_head_mask_opacity_strong)
      train_repair "$job" "$gpu" "$ALL20" 1200 "[300,600,900,1200]" \
        "++opt.lambda_silhouette_head_outer_shell=0.125" \
        "++opt.lambda_silhouette_outer_shell=0.042" \
        "++opt.boundary_aware_opacity_scale=0.34" \
        "++opt.boundary_aware_scaling_scale=0.12" \
        "++opt.boundary_aware_boundary_opacity_residual_scale=1.10" \
        "++opt.boundary_opacity_residual_lr=0.000070" \
        "++opt.boundary_scaling_residual_lr=0.000018" \
        "opt.opacity_lr=0.00022" \
        "opt.scaling_lr=0.000012" \
        "opt.grad_clip=0.0042"
      ;;
    v225c_testview_direct_diagnostic)
      train_repair "$job" "$gpu" "$HEAD_DIAG" 900 "[300,600,900]" \
        "++opt.train_sample_camera_weights={18:0.75,19:0.85,20:0.95,21:1.70,22:1.70,23:1.70}" \
        "++opt.train_sample_camera_min_prob=0.060" \
        "++opt.train_sample_camera_max_prob=0.260" \
        "++opt.lambda_silhouette_head_outer_shell=0.160" \
        "++opt.lambda_silhouette_outer_shell=0.046" \
        "++opt.lambda_silhouette_outer_spike=0.032" \
        "++opt.stageB_semantic_loss_enable=false" \
        "++opt.boundary_tag_topk_ratio=0.16" \
        "++opt.boundary_tag_min_ratio=0.10" \
        "++opt.boundary_aware_opacity_scale=0.46" \
        "++opt.boundary_aware_scaling_scale=0.18" \
        "++opt.boundary_aware_boundary_opacity_residual_scale=1.24" \
        "++opt.boundary_aware_boundary_scaling_residual_scale=0.60" \
        "++opt.boundary_opacity_residual_lr=0.000095" \
        "++opt.boundary_scaling_residual_lr=0.000026" \
        "opt.opacity_lr=0.00042" \
        "opt.scaling_lr=0.000028" \
        "opt.grad_clip=0.0050"
      ;;
    v225d_head_mask_xyz_probe)
      train_repair "$job" "$gpu" "$RELIABLE_PLUS" 900 "[300,600,900]" \
        "++opt.lambda_silhouette_head_outer_shell=0.115" \
        "++opt.boundary_aware_xyz_scale=0.035" \
        "++opt.boundary_aware_opacity_scale=0.30" \
        "++opt.boundary_aware_scaling_scale=0.12" \
        "opt.position_lr_init=0.000004" \
        "opt.position_lr_final=0.000001" \
        "opt.opacity_lr=0.00016" \
        "opt.scaling_lr=0.000010" \
        "opt.grad_clip=0.0038"
      ;;
    *)
      log_event "$gpu" "$job" "unknown_job" "skipped"
      return 2
      ;;
  esac
}

queue_gpu() {
  local gpu="$1"
  shift
  local jobs=("$@")
  local launched=0
  for job in "${jobs[@]}"; do
    run_named_job "$gpu" "$job"
    local status=$?
    launched=$((launched + 1))
    if [ "$status" -eq 0 ]; then
      log_event "$gpu" "$job" "job_done" "ok"
    else
      log_event "$gpu" "$job" "job_failed" "status=$status"
    fi
  done
  write_status "$gpu" "queue_done" "launched=$launched"
  log_event "$gpu" "queue" "done" "launched=$launched"
}

launch_queue() {
  local gpu="$1"
  local delay="${2:-0}"
  shift 2
  local jobs=("$@")
  (
    if [ "$delay" -gt 0 ]; then
      log_event "$gpu" "queue" "launch_delay" "${delay}s"
      sleep "$delay"
    fi
    queue_gpu "$gpu" "${jobs[@]}"
  ) &
  local pid=$!
  printf 'gpu%s_queue\t%s\t%s\n' "$gpu" "$gpu" "$pid" >> "$PIDS"
}

build_compare_panels() {
  if [ "$DO_RENDER" != "1" ] || [ "$SMOKE" = "1" ]; then
    return 0
  fi
  "$PYTHON_BIN" - "$SUMMARY" "$DATA_ROOT" "$LOG_DIR" "$BASE_RENDER" <<'PY' > "$LOG_DIR/build_compare_panels.log" 2>&1
import csv
import subprocess
import sys
from pathlib import Path

summary = Path(sys.argv[1])
data_root = Path(sys.argv[2])
log_dir = Path(sys.argv[3])
baseline = Path(sys.argv[4])
rows = list(csv.DictReader(summary.open(), delimiter="\t"))
render_rows = [
    row for row in rows
    if row.get("status") == "ok"
    and row.get("render_exp")
    and (Path(row["render_exp"]) / "test-view" / "renders").exists()
]
render_exps = [str(baseline)] if (baseline / "test-view" / "renders").exists() else []
labels = ["v224c_parserhard"]
for row in render_rows:
    render_exps.append(row["render_exp"])
    labels.append((row["name"] + "_" + row["kind"]).replace("v225", "225")[:28])
if len(render_exps) < 2:
    raise SystemExit(0)
cmd = [
    sys.executable, "tools/make_377_render_comparison_montage.py",
    "--render-exp", *render_exps,
    "--labels", *labels,
    "--gt-root", str(data_root / "CoreView_377"),
    "--output-dir", str(log_dir / "compare_panels" / "v225_head_silhouette"),
    "--select", "render_c21_f000240.png", "render_c21_f000300.png", "render_c22_f000240.png", "render_c23_f000300.png", "render_c23_f000420.png",
    "--crop", "150", "35", "650", "430",
    "--panel-width", "210",
    "--stack",
]
subprocess.run(cmd, check=False)
PY
}

mapfile -t SELECTED_GPUS < <(split_csv "$GPUS")
if [ "${#SELECTED_GPUS[@]}" -lt 1 ]; then
  echo "no GPUs selected in GPUS=$GPUS" >&2
  exit 4
fi

if [ "${#SELECTED_GPUS[@]}" -ge 1 ]; then
  launch_queue "${SELECTED_GPUS[0]}" 0 v225a_head_mask_opacity_moderate
fi
if [ "${#SELECTED_GPUS[@]}" -ge 2 ]; then
  launch_queue "${SELECTED_GPUS[1]}" "$QUEUE_LAUNCH_STAGGER_SECONDS" v225b_head_mask_opacity_strong
fi
if [ "${#SELECTED_GPUS[@]}" -ge 3 ] && [ "$RUN_DIAGNOSTIC_TESTVIEW" = "1" ]; then
  launch_queue "${SELECTED_GPUS[2]}" "$((QUEUE_LAUNCH_STAGGER_SECONDS * 2))" v225c_testview_direct_diagnostic
elif [ "${#SELECTED_GPUS[@]}" -ge 3 ]; then
  launch_queue "${SELECTED_GPUS[2]}" "$((QUEUE_LAUNCH_STAGGER_SECONDS * 2))" v225d_head_mask_xyz_probe
fi
if [ "${#SELECTED_GPUS[@]}" -ge 4 ]; then
  launch_queue "${SELECTED_GPUS[3]}" "$((QUEUE_LAUNCH_STAGGER_SECONDS * 3))" v225d_head_mask_xyz_probe
fi

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
cat "$PIDS"

wait
build_compare_panels
log_event "all" "queue" "all_done" "summary=$SUMMARY"
write_status "all" "all_done" "$SUMMARY"
