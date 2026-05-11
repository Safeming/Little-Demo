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
RUN_ID="${RUN_ID:-stageB_headfix_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SEED="${SEED:--1}"
GPUS="${GPUS:-0,1,2,3}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-2500}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-40}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
QUEUE_LAUNCH_STAGGER_SECONDS="${QUEUE_LAUNCH_STAGGER_SECONDS:-45}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
SMOKE="${SMOKE:-0}"
DO_RENDER="${DO_RENDER:-1}"
RUN_STRICT_BASELINE="${RUN_STRICT_BASELINE:-1}"
SINGLE_JOB="${SINGLE_JOB:-}"
SINGLE_GPU="${SINGLE_GPU:-auto}"

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
V223A_CKPT="${V223A_CKPT:-$V223A_EXP/best_ckpt.pth}"
V223A_RENDER="${V223A_RENDER:-${V223A_EXP}_render_full_best}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_head_silhouette_repair_4gpu_$RUN_ID}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
PIDS="$LOG_DIR/pids.tsv"
STATUS_JSON="$LOG_DIR/status.json"
mkdir -p "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$V223A_EXP/.hydra/config.yaml" "$V223A_CKPT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
CORE12="[2,3,6,7,8,9,11,13,15,18,19,20]"
RELIABLE8="[2,3,6,8,11,13,19,20]"
RELIABLE5="[2,8,11,19,20]"
HEAD_SELECT=(render_c21_f000240.png render_c21_f000300.png render_c22_f000240.png render_c23_f000300.png render_c23_f000420.png)
BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin]"
START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"

printf 'time_bjt\tgpu\tname\tphase\tdetail\n' > "$EVENTS"
printf 'name\tgpu\tpid\n' > "$PIDS"
printf 'name\tkind\texp_dir\trender_exp\tstatus\tdetail\n' > "$SUMMARY"
printf 'RUN_ID=%s\nSTART_BJT=%s\nGPUS=%s\nGPU_MAX_USED_MB_START=%s\nGPU_MAX_UTIL_START=%s\nWAIT_FOR_FREE_GPUS=%s\nV223A_EXP=%s\nV223A_CKPT=%s\nV223A_RENDER=%s\nDATA_ROOT=%s\nPARSER_ROOT=%s\nCOMPACT_MAPPING=%s\nSMOKE=%s\nDO_RENDER=%s\n' \
  "$RUN_ID" "$START_BJT" "$GPUS" "$GPU_MAX_USED_MB_START" "$GPU_MAX_UTIL_START" "$WAIT_FOR_FREE_GPUS" \
  "$V223A_EXP" "$V223A_CKPT" "$V223A_RENDER" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$SMOKE" "$DO_RENDER" | tee "$LOG_DIR/run_info.txt"

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

READY_GPU=""
wait_for_any_gpu() {
  local name="$1"
  READY_GPU=""
  if [ "$WAIT_FOR_FREE_GPUS" != "1" ] || [ "$SMOKE" = "1" ]; then
    READY_GPU="${SELECTED_GPUS[0]}"
    return 0
  fi

  local gpu used util detail
  while true; do
    detail=""
    for gpu in "${SELECTED_GPUS[@]}"; do
      read -r used util < <(gpu_stats "$gpu")
      used="${used:-999999}"
      util="${util:-100}"
      detail+="${gpu}:${used}MiB/${util}% "
      if [ "$used" -le "$GPU_MAX_USED_MB_START" ] && [ "$util" -le "$GPU_MAX_UTIL_START" ]; then
        READY_GPU="$gpu"
        log_event "$gpu" "$name" "gpu_ready_any" "used=${used}MiB util=${util}%"
        return 0
      fi
    done
    log_event "auto" "$name" "gpu_wait_any" "${detail}threshold=${GPU_MAX_USED_MB_START}MiB/${GPU_MAX_UTIL_START}%"
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
  shift 6
  local extra_overrides=("$@")
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_${label}"

  wait_for_gpu "$gpu" "${name}_render"
  log_event "$gpu" "$name" "render_start" "$out_exp"
  write_status "$gpu" "render_start" "$name"
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
    "++binding_map_mask_erode_kernel=1" \
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
    wandb_disable=true \
    "${extra_overrides[@]}" > "$LOG_DIR/${name}_render_${label}.log" 2>&1
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

train_headfix() {
  local name="$1"
  local gpu="$2"
  local train_views="$3"
  local iterations="$4"
  local checkpoint_list="$5"
  shift 5
  local overrides=("$@")
  if [ "$SMOKE" = "1" ]; then
    iterations=2
    checkpoint_list="[2]"
  fi

  local exp_dir="$ROOT/exp/stageB/377_hulk_light_${name}_${RUN_ID}"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_train"
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
    --config-path "$V223A_EXP/.hydra" \
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
    "start_checkpoint=$V223A_CKPT" \
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
    "++resume.clear_boundary_tags_on_resume=false" \
    "pipeline.pose_noise=0.0" \
    "model.gaussian.delay=0" \
    "++model.gaussian.semantic_logits_adapter_enable=true" \
    "++model.gaussian.semantic_logits_adapter_compact_classes=6" \
    "++model.pose_correction.train_root_orient=false" \
    "++model.pose_correction.train_pose_body=false" \
    "++model.pose_correction.train_pose_hand=false" \
    "++model.pose_correction.train_trans=false" \
    "++model.pose_correction.train_betas=false" \
    "opt.iterations=$iterations" \
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
    "++opt.texture_trainable_name_patterns=[__freeze_texture_no_match__]" \
    "++opt.camera_affine_enable=false" \
    "++opt.camera_affine_lr=0.0" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.semantic_region_logits_lr=0.0005" \
    "++opt.semantic_compact_logits_lr=0.0005" \
    "++opt.lambda_binding_semantic_adapter_reg=0.0015" \
    "++opt.stageB_semantic_loss_enable=true" \
    "++opt.stageB_semantic_ignore_uncertain=true" \
    "++opt.stageB_semantic_ignore_boundary_width=9" \
    "++opt.stageB_semantic_use_opacity_support=true" \
    "++opt.stageB_semantic_opacity_threshold=0.04" \
    "++opt.stageB_semantic_min_valid_pixels=96" \
    "++opt.stageB_semantic_body_cloth_weight=0.42" \
    "++opt.stageB_semantic_compact_weight=0.42" \
    "++opt.stageB_semantic_parent_consistency_weight=0.26" \
    "++opt.stageB_semantic_exclusive_weight=0.08" \
    "++opt.stageB_semantic_adapter_smooth_weight=0.018" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_min_prob=0.015" \
    "++opt.train_sample_camera_max_prob=0.105" \
    "++opt.train_sample_log_interval=100" \
    "++opt.train_sample_accumulation_steps=4" \
    "++opt.face_region_source=union" \
    "++opt.face_region_parser_dilate=3" \
    "++opt.boundary_region_source=soft_alpha" \
    "++opt.boundary_target_mask_source=soft" \
    "++opt.boundary_band_width=11" \
    "opt.lambda_mask=0.006" \
    "++opt.mask_loss_type=l1" \
    "++opt.lambda_mask_boundary=0.012" \
    "++opt.lambda_mask_boundary_hard=0.006" \
    "++opt.lambda_silhouette_outer=0.012" \
    "++opt.lambda_silhouette_outer_shell=0.024" \
    "++opt.lambda_silhouette_head_outer_shell=0.040" \
    "++opt.lambda_silhouette_outer_spike=0.014" \
    "++opt.silhouette_outer_ring_width=9" \
    "++opt.silhouette_outer_shell_start_width=1" \
    "++opt.silhouette_outer_shell_end_width=21" \
    "++opt.silhouette_outer_shell_soft_weights=true" \
    "++opt.silhouette_outer_shell_weight_min=0.35" \
    "++opt.silhouette_head_outer_region_dilate=21" \
    "++opt.silhouette_head_outer_bottom_ratio=0.36" \
    "++opt.silhouette_head_outer_fg_clip_dilate=39" \
    "++opt.silhouette_head_outer_min_pixels=16" \
    "++opt.boundary_aware_enable=true" \
    "++opt.boundary_aware_gate_l1_boundary=true" \
    "++opt.boundary_aware_gate_mask_boundary=true" \
    "++opt.boundary_aware_gate_mask_boundary_hard=true" \
    "++opt.boundary_aware_gate_silhouette_outer=true" \
    "++opt.boundary_aware_gate_silhouette_outer_shell=true" \
    "++opt.boundary_aware_gate_silhouette_head_outer_shell=true" \
    "++opt.boundary_aware_gate_silhouette_outer_spike=true" \
    "++opt.boundary_aware_threshold=0.032" \
    "++opt.boundary_aware_opacity_scale=0.0" \
    "++opt.boundary_aware_scaling_scale=0.0" \
    "++opt.boundary_aware_boundary_opacity_residual_scale=0.54" \
    "++opt.boundary_aware_boundary_scaling_residual_scale=0.30" \
    "++opt.boundary_opacity_residual_lr=1.5e-05" \
    "++opt.boundary_scaling_residual_lr=4.5e-06" \
    "++opt.lambda_boundary_opacity_residual_reg=0.0028" \
    "++opt.lambda_boundary_scaling_residual_reg=0.0010" \
    "++opt.lambda_boundary_opacity_residual_smooth=0.0024" \
    "++opt.lambda_boundary_scaling_residual_smooth=0.0018" \
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
    "opt.grad_clip=0.0025" \
    "${overrides[@]}" > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    summary_row "$name" "train" "$exp_dir" "" "train_failed" "status=$status"
    log_event "$gpu" "$name" "train_failed" "status=$status"
    return "$status"
  fi
  summary_row "$name" "train" "$exp_dir" "" "ok" "$exp_dir/best_ckpt.pth"
  log_event "$gpu" "$name" "train_done" "$LOG_DIR/${name}.log"
  if [ "$DO_RENDER" = "1" ] && [ "$SMOKE" != "1" ]; then
    render_export "$name" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "${exp_dir}_render_full_best" "best"
  fi
}

run_named_job() {
  local gpu="$1"
  local job="$2"
  case "$job" in
    baseline_v223a_strict_maps)
      render_export "$job" "$gpu" "$V223A_EXP" "$V223A_CKPT" "$ROOT/exp/stageB/377_hulk_light_v223a_strict_maps_${RUN_ID}" "strict_maps"
      ;;
    v224a_head_outer_mild)
      train_headfix "$job" "$gpu" "$ALL20" 1200 "[300,600,900,1200]" \
        "++opt.lambda_silhouette_head_outer_shell=0.034" \
        "++opt.lambda_silhouette_outer_shell=0.020" \
        "++opt.boundary_aware_boundary_opacity_residual_scale=0.48" \
        "++opt.boundary_opacity_residual_lr=1.1e-05" \
        "opt.grad_clip=0.0022"
      ;;
    v224b_head_outer_strong)
      train_headfix "$job" "$gpu" "$ALL20" 1200 "[300,600,900,1200]" \
        "++opt.lambda_silhouette_head_outer_shell=0.060" \
        "++opt.lambda_silhouette_outer_shell=0.032" \
        "++opt.lambda_silhouette_outer_spike=0.020" \
        "++opt.boundary_aware_boundary_opacity_residual_scale=0.66" \
        "++opt.boundary_aware_boundary_scaling_residual_scale=0.36" \
        "++opt.boundary_opacity_residual_lr=2.0e-05" \
        "++opt.boundary_scaling_residual_lr=6.0e-06" \
        "opt.grad_clip=0.0028"
      ;;
    v224c_head_reliable_views_preserve)
      train_headfix "$job" "$gpu" "$RELIABLE8" 1500 "[300,600,900,1200,1500]" \
        "++opt.train_sample_camera_weights={2:1.80,3:1.10,6:1.20,8:1.45,11:1.85,13:0.95,19:1.35,20:1.45}" \
        "++opt.train_sample_camera_min_prob=0.030" \
        "++opt.train_sample_camera_max_prob=0.180" \
        "++opt.lambda_silhouette_head_outer_shell=0.046" \
        "++opt.lambda_silhouette_outer_shell=0.024" \
        "++opt.semantic_region_logits_lr=0.00035" \
        "++opt.semantic_compact_logits_lr=0.00035" \
        "++opt.stageB_semantic_compact_weight=0.52" \
        "++opt.boundary_opacity_residual_lr=1.4e-05" \
        "opt.grad_clip=0.0024"
      ;;
    v224d_head_global_opacity_micro)
      train_headfix "$job" "$gpu" "$CORE12" 1000 "[250,500,750,1000]" \
        "opt.opacity_lr=2.0e-05" \
        "opt.scaling_lr=1.0e-06" \
        "++opt.lambda_silhouette_head_outer_shell=0.052" \
        "++opt.lambda_silhouette_outer_shell=0.026" \
        "++opt.boundary_aware_opacity_scale=0.04" \
        "++opt.boundary_aware_scaling_scale=0.02" \
        "++opt.boundary_aware_boundary_opacity_residual_scale=0.50" \
        "++opt.boundary_opacity_residual_lr=1.2e-05" \
        "++opt.boundary_scaling_residual_lr=3.8e-06" \
        "opt.grad_clip=0.0018"
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
  "$PYTHON_BIN" - "$SUMMARY" "$DATA_ROOT" "$LOG_DIR" "$V223A_RENDER" <<'PY' > "$LOG_DIR/build_compare_panels.log" 2>&1
import csv
import subprocess
import sys
from pathlib import Path
summary = Path(sys.argv[1])
data_root = Path(sys.argv[2])
log_dir = Path(sys.argv[3])
baseline = Path(sys.argv[4])
rows = list(csv.DictReader(summary.open(), delimiter="\t"))
render_rows = [row for row in rows if row.get("status") == "ok" and row.get("render_exp") and (Path(row["render_exp"]) / "test-view" / "renders").exists()]
render_exps = [str(baseline)] if (baseline / "test-view" / "renders").exists() else []
labels = ["v223a"]
for row in render_rows:
    name = row["name"]
    if name == "baseline_v223a_strict_maps":
        continue
    render_exps.append(row["render_exp"])
    labels.append(name.replace("v224", "224")[:24])
if len(render_exps) < 2:
    raise SystemExit(0)
cmd = [
    sys.executable, "tools/make_377_render_comparison_montage.py",
    "--render-exp", *render_exps,
    "--labels", *labels,
    "--gt-root", str(data_root / "CoreView_377"),
    "--output-dir", str(log_dir / "compare_panels" / "headfix"),
    "--select", "render_c21_f000240.png", "render_c21_f000300.png", "render_c22_f000240.png", "render_c23_f000300.png", "render_c23_f000420.png",
    "--crop", "150", "35", "650", "430",
    "--panel-width", "210",
    "--stack",
]
subprocess.run(cmd, check=False)
PY
}

mapfile -t SELECTED_GPUS < <(split_csv "$GPUS")
if [ "${#SELECTED_GPUS[@]}" -lt 4 ]; then
  echo "expected four GPUs in GPUS, got: $GPUS" >&2
  exit 4
fi

GPU0="${SELECTED_GPUS[0]}"
GPU1="${SELECTED_GPUS[1]}"
GPU2="${SELECTED_GPUS[2]}"
GPU3="${SELECTED_GPUS[3]}"

if [ -n "$SINGLE_JOB" ]; then
  if [ "$SINGLE_GPU" = "auto" ]; then
    wait_for_any_gpu "$SINGLE_JOB"
    SINGLE_GPU="$READY_GPU"
  fi
  queue_gpu "$SINGLE_GPU" "$SINGLE_JOB"
  build_compare_panels
  log_event "all" "queue" "all_done" "summary=$SUMMARY"
  write_status "all" "all_done" "$SUMMARY"
  exit 0
fi

if [ "$RUN_STRICT_BASELINE" = "1" ]; then
  launch_queue "$GPU0" 0 baseline_v223a_strict_maps v224a_head_outer_mild
else
  launch_queue "$GPU0" 0 v224a_head_outer_mild
fi
launch_queue "$GPU1" "$QUEUE_LAUNCH_STAGGER_SECONDS" v224b_head_outer_strong
launch_queue "$GPU2" "$((QUEUE_LAUNCH_STAGGER_SECONDS * 2))" v224c_head_reliable_views_preserve
launch_queue "$GPU3" "$((QUEUE_LAUNCH_STAGGER_SECONDS * 3))" v224d_head_global_opacity_micro

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
cat "$PIDS"

wait
build_compare_panels
log_event "all" "queue" "all_done" "summary=$SUMMARY"
write_status "all" "all_done" "$SUMMARY"
