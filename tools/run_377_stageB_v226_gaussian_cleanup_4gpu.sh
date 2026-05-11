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
RUN_ID="${RUN_ID:-stageB_v226_cleanup_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
GPUS="${GPUS:-0,1,2,3}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-2500}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-45}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
SMOKE="${SMOKE:-0}"
DO_RECOVERY_TRAIN="${DO_RECOVERY_TRAIN:-1}"
DO_RGBCLIP_EXPORT="${DO_RGBCLIP_EXPORT:-1}"

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
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v224c_head_reliable_views_preserve_stageB_headfix_fixed_20260511_103654_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/best_ckpt.pth}"
BASE_RENDER="${BASE_RENDER:-${BASE_EXP}_render_parserhard_best}"
BASE_ITER="${BASE_ITER:-111710}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v226_gaussian_cleanup_$RUN_ID}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
PIDS="$LOG_DIR/pids.tsv"
STATUS_JSON="$LOG_DIR/status.json"
HEAD_SELECT=(render_c21_f000240.png render_c21_f000300.png render_c22_f000240.png render_c23_f000300.png render_c23_f000420.png)
BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin]"
ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"

mkdir -p "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

IFS=',' read -ra SELECTED_GPUS <<< "$GPUS"
if [ "${#SELECTED_GPUS[@]}" -lt 1 ]; then
  echo "no GPUs selected" >&2
  exit 2
fi

printf 'time_bjt\tgpu\tname\tphase\tdetail\n' > "$EVENTS"
printf 'name\tgpu\tpid\n' > "$PIDS"
printf 'name\tkind\texp_dir\trender_exp\tstatus\tdetail\n' > "$SUMMARY"
printf 'RUN_ID=%s\nSTART_BJT=%s\nGPUS=%s\nBASE_EXP=%s\nBASE_CKPT=%s\nBASE_RENDER=%s\nBASE_ITER=%s\nDATA_ROOT=%s\nPARSER_ROOT=%s\nCOMPACT_MAPPING=%s\nSMOKE=%s\nDO_RECOVERY_TRAIN=%s\nDO_RGBCLIP_EXPORT=%s\n' \
  "$RUN_ID" "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$GPUS" "$BASE_EXP" "$BASE_CKPT" "$BASE_RENDER" "$BASE_ITER" \
  "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$SMOKE" "$DO_RECOVERY_TRAIN" "$DO_RGBCLIP_EXPORT" | tee "$LOG_DIR/run_info.txt"

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
    --baseline-render-exp "$BASE_RENDER" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --select "${HEAD_SELECT[@]}" \
    --out-dir "$render_exp/diagnostics/head_silhouette_${label}" \
    --panels gt gt_mask render outside_gt base_render base_diff layer region compact_semantic semantic \
    > "$LOG_DIR/${name}_head_diag_${label}.log" 2>&1 || true
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
  run_head_diag "$name" "$gpu" "$out_exp" "$label"
  summary_row "$name" "$label" "$config_exp" "$out_exp" "ok" "$ckpt"
  log_event "$gpu" "$name" "render_done" "$out_exp"
}

make_compare_panel() {
  local out_dir="$LOG_DIR/compare_panels/v226_gaussian_cleanup"
  mkdir -p "$out_dir"
  "$PYTHON_BIN" tools/make_377_render_comparison_montage.py \
    --render-exp \
      "$BASE_RENDER" \
      "$ROOT/exp/stageB/377_hulk_light_v226b_opacity_cap_mild_${RUN_ID}_render_parserhard_best" \
      "$ROOT/exp/stageB/377_hulk_light_v226c_opacity_cap_strong_${RUN_ID}_render_parserhard_best" \
      "$ROOT/exp/stageB/377_hulk_light_v226d_prune_highconf_${RUN_ID}_render_parserhard_best" \
    --labels v224c v226b_cap_mild v226c_cap_strong v226d_prune \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --split test-view \
    --output-dir "$out_dir" \
    --select "${HEAD_SELECT[@]}" \
    --panel-width 220 \
    --header-height 34 \
    --stack > "$LOG_DIR/build_compare_panels.log" 2>&1 || true
}

train_recovery() {
  local gpu="$1"
  local start_exp="$2"
  local start_ckpt="$3"
  local name="v226e_prune_recover_short"
  local iterations=600
  local checkpoint_list="[300,600]"
  if [ "$SMOKE" = "1" ]; then
    iterations=4
    checkpoint_list="[2,4]"
  fi
  local exp_dir="$ROOT/exp/stageB/377_hulk_light_${name}_${RUN_ID}"
  local final_ckpt="$exp_dir/ckpt$((BASE_ITER + iterations)).pth"

  if [ ! -f "$start_ckpt" ]; then
    summary_row "$name" "train" "$exp_dir" "" "train_skipped" "missing_start_ckpt=$start_ckpt"
    return 0
  fi

  wait_for_gpu "$gpu" "$name"
  log_event "$gpu" "$name" "train_start" "start=$start_ckpt iterations=$iterations"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" train.py \
    --config-path "$start_exp/.hydra" \
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
    "start_checkpoint=$start_ckpt" \
    "exp_dir=$exp_dir" \
    "seed=-1" \
    "wandb_disable=true" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/${name}_train" \
    "++resume.allow_partial_converter_load=true" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.,pose_correction.pose_body_train_mask]" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=true" \
    "++resume.use_checkpoint_iteration_as_offset=true" \
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
    "++opt.semantic_region_logits_lr=0.00012" \
    "++opt.semantic_compact_logits_lr=0.00012" \
    "++opt.stageB_semantic_loss_enable=true" \
    "++opt.stageB_semantic_ignore_uncertain=true" \
    "++opt.stageB_semantic_ignore_boundary_width=11" \
    "++opt.stageB_semantic_body_cloth_weight=0.30" \
    "++opt.stageB_semantic_compact_weight=0.36" \
    "++opt.stageB_semantic_parent_consistency_weight=0.18" \
    "++opt.stageB_semantic_adapter_smooth_weight=0.018" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_accumulation_steps=4" \
    "++opt.foreground_mask_source=hard" \
    "++opt.global_mask_source=hard" \
    "++opt.boundary_target_mask_source=hard" \
    "++opt.face_region_source=union" \
    "++opt.boundary_region_source=binary" \
    "++opt.boundary_band_width=9" \
    "opt.lambda_mask=0.006" \
    "++opt.mask_loss_type=l1" \
    "++opt.lambda_mask_boundary=0.012" \
    "++opt.lambda_mask_boundary_hard=0.008" \
    "++opt.lambda_silhouette_outer=0.010" \
    "++opt.lambda_silhouette_outer_shell=0.022" \
    "++opt.lambda_silhouette_head_outer_shell=0.055" \
    "++opt.silhouette_outer_ring_width=7" \
    "++opt.silhouette_outer_shell_start_width=1" \
    "++opt.silhouette_outer_shell_end_width=21" \
    "++opt.silhouette_head_outer_region_dilate=25" \
    "++opt.silhouette_head_outer_bottom_ratio=0.42" \
    "++opt.boundary_aware_enable=true" \
    "++opt.boundary_aware_gate_mask_boundary=true" \
    "++opt.boundary_aware_gate_silhouette_outer=true" \
    "++opt.boundary_aware_gate_silhouette_outer_shell=true" \
    "++opt.boundary_aware_gate_silhouette_head_outer_shell=true" \
    "++opt.boundary_aware_threshold=0.018" \
    "++opt.boundary_aware_freeze_converter_for_boundary_loss=true" \
    "++opt.boundary_aware_feature_dc_scale=0.0" \
    "++opt.boundary_aware_feature_rest_scale=0.0" \
    "++opt.boundary_aware_xyz_scale=0.0" \
    "++opt.boundary_aware_opacity_scale=0.12" \
    "++opt.boundary_aware_scaling_scale=0.04" \
    "++opt.boundary_opacity_residual_lr=0.000024" \
    "++opt.boundary_scaling_residual_lr=0.000006" \
    "opt.opacity_lr=0.000055" \
    "opt.scaling_lr=0.000003" \
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
    "opt.grad_clip=0.0025" > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    summary_row "$name" "train" "$exp_dir" "" "train_failed" "status=$status"
    log_event "$gpu" "$name" "train_failed" "status=$status"
    return "$status"
  fi
  summary_row "$name" "train" "$exp_dir" "" "ok" "best=$exp_dir/best_ckpt.pth final=$final_ckpt"
  log_event "$gpu" "$name" "train_done" "$LOG_DIR/${name}.log"
  render_export "$name" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "${exp_dir}_render_parserhard_best" "best"
  render_export "$name" "$gpu" "$exp_dir" "$final_ckpt" "${exp_dir}_render_parserhard_final" "final"
}

run_cleanup_generation() {
  local gpu="${SELECTED_GPUS[0]}"
  wait_for_gpu "$gpu" "v226_cleanup_generation"
  log_event "$gpu" "v226_cleanup_generation" "start" "$RUN_ID"
  write_status "$gpu" "cleanup_generation" "$RUN_ID"
  local extra=()
  if [ "$SMOKE" = "1" ]; then
    extra+=(--write-report-only)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/make_377_stageB_v226_cleanup_ckpts.py \
    --base-exp "$BASE_EXP" \
    --base-ckpt "$BASE_CKPT" \
    --out-root "$ROOT/exp/stageB" \
    --run-id "$RUN_ID" \
    --data-root "$DATA_ROOT" \
    --parser-root "$PARSER_ROOT" \
    --compact-mapping "$COMPACT_MAPPING" \
    --base-iter "$BASE_ITER" \
    "${extra[@]}" > "$LOG_DIR/make_cleanup_ckpts.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    summary_row "v226_cleanup_generation" "make_ckpts" "$BASE_EXP" "" "failed" "status=$status"
    log_event "$gpu" "v226_cleanup_generation" "failed" "status=$status"
    return "$status"
  fi
  summary_row "v226a_candidate_report" "report" "$ROOT/exp/stageB/377_hulk_light_v226a_candidate_report_${RUN_ID}" "" "ok" "$LOG_DIR/make_cleanup_ckpts.log"
  log_event "$gpu" "v226_cleanup_generation" "done" "$LOG_DIR/make_cleanup_ckpts.log"
}

run_render_queue() {
  local queue_id="$1"
  local gpu="$2"
  shift 2
  local jobs=("$@")
  local job exp ckpt out_exp
  for job in "${jobs[@]}"; do
    exp="$ROOT/exp/stageB/377_hulk_light_${job}_${RUN_ID}"
    ckpt="$exp/best_ckpt.pth"
    out_exp="${exp}_render_parserhard_best"
    render_export "$job" "$gpu" "$exp" "$ckpt" "$out_exp" "best"
    if [ "$DO_RGBCLIP_EXPORT" = "1" ]; then
      render_export "$job" "$gpu" "$exp" "$ckpt" "${exp}_render_rgbclip_best" "rgbclip_best" \
        "++render_export_mask_source=parser_hard" \
        "++render_export_mask_erode_kernel=3" \
        "++render_export_fill_close_kernel=1" \
        "++render_export_clip_to_mask=true" \
        "++render_export_fill_dilate_kernel=1"
    fi
  done
  log_event "$gpu" "$queue_id" "done" "jobs=${jobs[*]}"
}

run_cleanup_generation || exit 1

if [ "$SMOKE" = "1" ]; then
  log_event "all" "smoke" "done" "generation only"
  exit 0
fi

GPU0="${SELECTED_GPUS[0]}"
GPU1="${SELECTED_GPUS[1]:-${SELECTED_GPUS[0]}}"
GPU2="${SELECTED_GPUS[2]:-${SELECTED_GPUS[0]}}"
GPU3="${SELECTED_GPUS[3]:-${SELECTED_GPUS[0]}}"

run_render_queue queue_v226b "$GPU0" v226b_opacity_cap_mild &
printf 'queue_v226b\t%s\t%s\n' "$GPU0" "$!" >> "$PIDS"
run_render_queue queue_v226c "$GPU1" v226c_opacity_cap_strong &
printf 'queue_v226c\t%s\t%s\n' "$GPU1" "$!" >> "$PIDS"
run_render_queue queue_v226d "$GPU2" v226d_prune_highconf &
printf 'queue_v226d\t%s\t%s\n' "$GPU2" "$!" >> "$PIDS"

if [ "$DO_RECOVERY_TRAIN" = "1" ]; then
  train_recovery "$GPU3" "$ROOT/exp/stageB/377_hulk_light_v226d_prune_highconf_${RUN_ID}" "$ROOT/exp/stageB/377_hulk_light_v226d_prune_highconf_${RUN_ID}/best_ckpt.pth" &
  printf 'v226e_prune_recover_short\t%s\t%s\n' "$GPU3" "$!" >> "$PIDS"
fi

wait
make_compare_panel
log_event "all" "queue" "all_done" "summary=$SUMMARY"
write_status "all" "all_done" "$SUMMARY"
