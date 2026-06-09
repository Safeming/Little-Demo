#!/usr/bin/env bash
set -u
set -o pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-stageB_v261_candidate_validator_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v261_candidate_validator_${RUN_ID}}"
mkdir -p "$LOG_DIR"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_shoes_preserve_control_stageB_compact_v233_skincloth_20260512_161652_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt135710.pth}"
BASE_ITER="${BASE_ITER:-135710}"
BASE_RENDER_EXP="${BASE_RENDER_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_stageC_softer_stageC_v235_v236_20260513_114036_bjt_render_compact_final}"

SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
STATUS_JSON="$LOG_DIR/status.json"
VALIDATOR_DIR="$LOG_DIR/v261_candidate_validator"
GATE_JSON="$LOG_DIR/v261_do_no_harm_gate.json"

RUN_SHORT_TRAIN="${RUN_SHORT_TRAIN:-0}"
DO_RENDER="${DO_RENDER:-1}"
WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-18000}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-65}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"

TRAIN_NAME="${TRAIN_NAME:-v261a_candidate_validated_support}"
RENDER_TAG="${RENDER_TAG:-v261}"
TRAIN_VIEWS="${TRAIN_VIEWS:-[1,2,3,4,5,6,7,8,9,10,11,12]}"
VAL_VIEWS="${VAL_VIEWS:-[21,22,23]}"
TEST_VIEWS="${TEST_VIEWS:-[21,22,23]}"
TRAIN_FRAMES="${TRAIN_FRAMES:-[0,570,1]}"
VAL_FRAMES="${VAL_FRAMES:-[0,570,60]}"
TEST_FRAMES="${TEST_FRAMES:-[0,570,60]}"
RENDER_TEST_VIEWS="${RENDER_TEST_VIEWS:-[21,22,23]}"
RENDER_TEST_FRAMES="${RENDER_TEST_FRAMES:-[0,570,60]}"
RENDER_SPLIT_DIR="${RENDER_SPLIT_DIR:-test-view}"

printf 'time_bjt\tgpu\tname\tphase\tdetail\n' > "$EVENTS"
printf 'name\tkind\texp_dir\trender_exp\tckpt\tstatus\tdetail\n' > "$SUMMARY"

log_event() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$GPU" "$1" "$2" "$3" | tee -a "$EVENTS"
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
      log_event "$name" "gpu_ready" "used=${used}MiB util=${util}%"
      return 0
    fi
    log_event "$name" "gpu_wait" "used=${used}MiB util=${util}% threshold=${GPU_MAX_USED_MB_START}MiB/${GPU_MAX_UTIL_START}%"
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
      log_event "preflight" "missing" "$required"
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    write_status "blocked" "missing required paths; see $EVENTS"
    return 2
  fi
}

run_baseline_metrics() {
  local residual_dir="$LOG_DIR/v233d_baseline_boundary_residuals"
  local contour_dir="$LOG_DIR/v233d_baseline_contours"
  log_event "v233d_baseline" "metrics_start" "$BASE_RENDER_EXP"
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_boundary_residuals.py" \
    --render-exp "$BASE_RENDER_EXP" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --search-band-width 24 \
    --out-dir "$residual_dir" \
    > "$LOG_DIR/v233d_baseline_residuals.log" 2>&1
  local residual_status=$?
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_render_contours.py" \
    --render-exp "$BASE_RENDER_EXP" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --out-dir "$contour_dir" \
    > "$LOG_DIR/v233d_baseline_contours.log" 2>&1
  local contour_status=$?
  if [ "$residual_status" -ne 0 ] || [ "$contour_status" -ne 0 ]; then
    log_event "v233d_baseline" "metrics_failed" "residual=$residual_status contour=$contour_status"
    return 2
  fi
  summary_row "v233d_baseline_residuals" "diagnostic" "" "$BASE_RENDER_EXP" "" "ok" "$residual_dir"
  summary_row "v233d_baseline_contours" "diagnostic" "" "$BASE_RENDER_EXP" "" "ok" "$contour_dir"
  log_event "v233d_baseline" "metrics_done" "residual=$residual_dir contour=$contour_dir"
}

run_candidate_validator() {
  wait_for_gpu "v261_candidate_validator"
  log_event "v261_candidate_validator" "start" "base_ckpt=$BASE_CKPT"
  common_env "$PYTHON_BIN" "$ROOT/tools/validate_377_stageB_v261_support_candidates.py" \
    --config-path "$BASE_EXP/.hydra/config.yaml" \
    --load-ckpt "$BASE_CKPT" \
    --out-dir "$VALIDATOR_DIR" \
    --dataset-root "$DATA_ROOT" \
    --parser-root "$PARSER_ROOT" \
    --compact-mapping "$COMPACT_MAPPING" \
    --candidate-views "${CANDIDATE_VIEWS:-1,2,3,4,5,6,7,8,9,10,11,12}" \
    --candidate-frames "${CANDIDATE_FRAMES:-0,570,60}" \
    --eval-views "${EVAL_VIEWS:-21,22,23}" \
    --eval-frames "${EVAL_FRAMES:-0,570,60}" \
    --iteration "$((BASE_ITER + 1))" \
    --max-candidate-points "${MAX_CANDIDATE_POINTS:-72}" \
    --min-accepted-candidates "${MIN_ACCEPTED_CANDIDATES:-2}" \
    --min-inner-outer-ratio "${MIN_INNER_OUTER_RATIO:-1.35}" \
    --min-inner-hit-pixels "${MIN_INNER_HIT_PIXELS:-12.0}" \
    > "$LOG_DIR/v261_candidate_validator.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "v261_candidate_validator" "failed" "status=$status"
    summary_row "v261_candidate_validator" "diagnostic" "" "" "$BASE_CKPT" "failed" "status=$status"
    return "$status"
  fi
  summary_row "v261_candidate_validator" "diagnostic" "" "" "$BASE_CKPT" "ok" "$VALIDATOR_DIR"
  log_event "v261_candidate_validator" "done" "$VALIDATOR_DIR/candidate_validation_summary.json"
}

run_pretrain_gate() {
  log_event "v261_gate" "start" "candidate=$VALIDATOR_DIR/candidate_validation_summary.json"
  "$PYTHON_BIN" "$ROOT/tools/check_377_stageB_v261_do_no_harm.py" \
    --candidate-summary "$VALIDATOR_DIR/candidate_validation_summary.json" \
    --require-candidate-ok \
    --out-json "$GATE_JSON" \
    > "$LOG_DIR/v261_do_no_harm_gate.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "v261_gate" "blocked" "$GATE_JSON"
    summary_row "v261_gate" "gate" "" "" "$BASE_CKPT" "blocked" "$GATE_JSON"
    return "$status"
  fi
  summary_row "v261_gate" "gate" "" "" "$BASE_CKPT" "ok" "$GATE_JSON"
  log_event "v261_gate" "ok" "$GATE_JSON"
}

render_checkpoint() {
  local exp_dir="$1"
  local local_step="$2"
  local global_iter=$((BASE_ITER + local_step))
  local ckpt="$exp_dir/ckpt${global_iter}.pth"
  local render_exp="${exp_dir}_render_${RENDER_TAG}_ckpt${global_iter}"
  if [ ! -f "$ckpt" ]; then
    log_event "$TRAIN_NAME" "render_skip" "missing=$ckpt"
    return 0
  fi
  wait_for_gpu "${TRAIN_NAME}_render_${global_iter}"
  log_event "$TRAIN_NAME" "render_start" "$ckpt"
  common_env "$PYTHON_BIN" "$ROOT/render.py" \
    --config-path "$exp_dir/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=$RENDER_TEST_VIEWS" \
    "dataset.test_frames.view=$RENDER_TEST_FRAMES" \
    "dataset.parsing_prior.enable=true" \
    "dataset.parsing_prior.roi_enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.parser_layout=cihp_subject" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING" \
    "export_interpretability=true" \
    "++binding_map_names=[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin,boundary_support]" \
    "++binding_map_use_opacity_mask=true" \
    "++binding_map_hard_fg_opacity_threshold=0.030" \
    "++binding_map_opacity_threshold=0.030" \
    "++binding_map_compact_semantic_opacity_threshold=0.025" \
    "++binding_map_hard_fg_close_kernel=5" \
    "++binding_map_hard_fg_erode_kernel=1" \
    "++render_export_opacity_threshold=0.025" \
    "++render_export_close_kernel=5" \
    "++render_export_erode_kernel=1" \
    "hydra.run.dir=$LOG_DIR/hydra_${TRAIN_NAME}_render_ckpt${global_iter}" \
    "wandb_disable=true" \
    > "$LOG_DIR/${TRAIN_NAME}_render_ckpt${global_iter}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "$TRAIN_NAME" "render_failed" "status=$status"
    summary_row "$TRAIN_NAME" "render" "$exp_dir" "$render_exp" "$ckpt" "failed" "status=$status"
    return "$status"
  fi
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_boundary_residuals.py" \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir "$RENDER_SPLIT_DIR" \
    --band-width 7 \
    --search-band-width 24 \
    --out-dir "$render_exp/diagnostics/${RENDER_SPLIT_DIR}_boundary_residuals" \
    > "$LOG_DIR/${TRAIN_NAME}_residuals_ckpt${global_iter}.log" 2>&1 || true
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_render_contours.py" \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir "$RENDER_SPLIT_DIR" \
    --band-width 7 \
    --out-dir "$render_exp/diagnostics/${RENDER_SPLIT_DIR}_contours" \
    > "$LOG_DIR/${TRAIN_NAME}_contours_ckpt${global_iter}.log" 2>&1 || true
  "$PYTHON_BIN" "$ROOT/tools/check_377_stageB_v261_do_no_harm.py" \
    --candidate-summary "$VALIDATOR_DIR/candidate_validation_summary.json" \
    --require-candidate-ok \
    --residual-summary "$render_exp/diagnostics/${RENDER_SPLIT_DIR}_boundary_residuals/boundary_residual_summary.json" \
    --contour-summary "$render_exp/diagnostics/${RENDER_SPLIT_DIR}_contours/contour_summary.json" \
    --out-json "$render_exp/diagnostics/${RENDER_SPLIT_DIR}_do_no_harm_gate.json" \
    > "$LOG_DIR/${TRAIN_NAME}_do_no_harm_ckpt${global_iter}.log" 2>&1 || true
  log_event "$TRAIN_NAME" "render_done" "$render_exp"
  summary_row "$TRAIN_NAME" "render" "$exp_dir" "$render_exp" "$ckpt" "ok" "local_step=$local_step"
}

run_short_train_if_enabled() {
  if [ "$RUN_SHORT_TRAIN" != "1" ]; then
    log_event "$TRAIN_NAME" "train_skipped" "RUN_SHORT_TRAIN=$RUN_SHORT_TRAIN validator_only"
    summary_row "$TRAIN_NAME" "train" "" "" "" "skipped" "RUN_SHORT_TRAIN=$RUN_SHORT_TRAIN"
    return 0
  fi

  local iterations="${STAGEB_ITERATIONS:-120}"
  local checkpoint_steps="${STAGEB_CHECKPOINT_STEPS:-120}"
  local checkpoint_list="[$checkpoint_steps]"
  local exp_dir="$ROOT/exp/stageB/377_hulk_light_${TRAIN_NAME}_${RUN_ID}"
  local final_ckpt="$exp_dir/ckpt$((BASE_ITER + iterations)).pth"
  mkdir -p "$exp_dir"

  wait_for_gpu "$TRAIN_NAME"
  log_event "$TRAIN_NAME" "train_start" "iterations=$iterations ckpt=$BASE_CKPT"
  common_env "$PYTHON_BIN" "$ROOT/train.py" \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$TRAIN_VIEWS" \
    "dataset.val_views=$VAL_VIEWS" \
    "dataset.test_views.view=$TEST_VIEWS" \
    "dataset.train_frames=$TRAIN_FRAMES" \
    "dataset.val_frames=$VAL_FRAMES" \
    "dataset.test_frames.view=$TEST_FRAMES" \
    "dataset.parsing_prior.enable=true" \
    "dataset.parsing_prior.roi_enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.parser_layout=cihp_subject" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING" \
    "start_checkpoint=$BASE_CKPT" \
    "exp_dir=$exp_dir" \
    "seed=${SEED:--1}" \
    "hydra.run.dir=$LOG_DIR/hydra_${TRAIN_NAME}_train" \
    "wandb_disable=true" \
    "++resume.allow_partial_converter_load=false" \
    "++resume.restore_gaussian_optimizer_state=false" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.disable_densify_on_resume=false" \
    "++resume.disable_opacity_reset_on_resume=true" \
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
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.semantic_region_logits_lr=0.0" \
    "++opt.semantic_compact_logits_lr=0.0" \
    "++opt.stageB_semantic_loss_enable=false" \
    "++opt.foreground_mask_source=hard" \
    "++opt.global_mask_source=hard" \
    "++opt.boundary_target_mask_source=hard" \
    "opt.lambda_mask=0.004" \
    "++opt.lambda_mask_boundary=0.006" \
    "++opt.lambda_mask_boundary_hard=0.004" \
    "++opt.lambda_silhouette_outer=0.006" \
    "++opt.lambda_silhouette_outer_shell=0.010" \
    "++opt.boundary_aware_enable=false" \
    "++opt.boundary_aware_xyz_scale=0.0" \
    "++opt.boundary_aware_opacity_scale=0.0" \
    "++opt.boundary_aware_scaling_scale=0.0" \
    "opt.position_lr_init=0.0" \
    "opt.position_lr_final=0.0" \
    "opt.opacity_lr=0.0" \
    "opt.scaling_lr=0.0" \
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "opt.lambda_l1=0.35" \
    "opt.lambda_l1_fg=0.65" \
    "opt.lambda_l1_boundary=0.22" \
    "opt.lambda_perceptual=0.004" \
    "++opt.lambda_detail_face=0.0" \
    "++opt.lambda_detail_shoulder_arm=0.0" \
    "++opt.lambda_detail_waist=0.0" \
    "opt.lambda_skinning=0.0" \
    "opt.lambda_aiap_xyz=0.0" \
    "opt.lambda_aiap_cov=0.0" \
    "test_interval=1000" \
    "test_iterations=$checkpoint_list" \
    "save_iterations=$checkpoint_list" \
    "checkpoint_iterations=$checkpoint_list" \
    "++validation_image_log_limit=0" \
    "opt.percent_dense=0.030" \
    "opt.densification_interval=${DENSIFY_INTERVAL:-120}" \
    "opt.densify_from_iter=$((BASE_ITER + ${DENSIFY_FROM_STEP:-20}))" \
    "opt.densify_until_iter=$((BASE_ITER + ${DENSIFY_UNTIL_STEP:-121}))" \
    "opt.densify_grad_threshold=${DENSIFY_GRAD_THRESHOLD:-0.00110}" \
    "opt.opacity_threshold=0.000001" \
    "opt.opacity_reset_interval=1000000" \
    "++model.gaussian.binding_densify_disable_clone=true" \
    "++model.gaussian.binding_densify_disable_split=true" \
    "++opt.boundary_component_support_enable=true" \
    "++opt.boundary_component_support_interval=20" \
    "++opt.boundary_component_support_residual_threshold=0.50" \
    "++opt.boundary_component_support_min_area=18" \
    "++opt.boundary_component_support_max_components=12" \
    "++opt.boundary_component_support_points_per_component=2" \
    "++opt.boundary_component_support_max_points_per_view=24" \
    "++opt.boundary_component_support_target_project_enable=true" \
    "++opt.boundary_component_support_target_project_offset_min=0.000001" \
    "++opt.boundary_component_support_target_project_offset_max=0.080" \
    "++model.gaussian.boundary_component_support_enable=true" \
    "++model.gaussian.boundary_component_support_accumulate_pending=true" \
    "++model.gaussian.boundary_component_support_pending_max_views=8" \
    "++model.gaussian.boundary_component_support_pending_max_points=192" \
    "++model.gaussian.boundary_component_support_candidate_consensus_enable=true" \
    "++model.gaussian.boundary_component_support_candidate_consensus_strict=true" \
    "++model.gaussian.boundary_component_support_candidate_consensus_min_votes=2" \
    "++model.gaussian.boundary_component_support_candidate_consensus_min_unique_views=2" \
    "++model.gaussian.boundary_component_support_candidate_consensus_cluster_radius=0.018" \
    "++model.gaussian.boundary_component_support_candidate_consensus_max_xyz_std=0.020" \
    "++model.gaussian.boundary_component_support_candidate_consensus_target_under_min=0.55" \
    "++model.gaussian.boundary_component_support_candidate_consensus_target_over_max=0.35" \
    "++model.gaussian.boundary_component_support_candidate_consensus_anchor_over_max=0.30" \
    "++model.gaussian.boundary_component_support_max_points=72" \
    "++model.gaussian.boundary_component_support_use_target_offsets=true" \
    "++model.gaussian.boundary_component_support_target_offset_max=0.080" \
    "++model.gaussian.boundary_component_support_child_opacity_factor=0.85" \
    "++model.gaussian.boundary_component_support_child_opacity_floor=0.035" \
    "++model.gaussian.boundary_component_support_child_opacity_ceiling=0.34" \
    "++model.gaussian.boundary_component_support_child_scale_factor=0.55" \
    "++model.gaussian.boundary_support_projector_enable=false" \
    "opt.grad_clip=0.0022" \
    > "$LOG_DIR/${TRAIN_NAME}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "$TRAIN_NAME" "train_failed" "status=$status"
    summary_row "$TRAIN_NAME" "train" "$exp_dir" "" "$final_ckpt" "failed" "status=$status"
    return "$status"
  fi
  summary_row "$TRAIN_NAME" "train" "$exp_dir" "" "$final_ckpt" "ok" "iterations=$iterations"
  log_event "$TRAIN_NAME" "train_done" "$final_ckpt"
  if [ "$DO_RENDER" = "1" ]; then
    render_checkpoint "$exp_dir" "$iterations" || true
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
  echo "RUN_SHORT_TRAIN=$RUN_SHORT_TRAIN"
} | tee "$LOG_DIR/run_info.txt"

write_status "starting" "preflight"
check_required || exit $?
run_baseline_metrics || exit $?
run_candidate_validator || exit $?
run_pretrain_gate || {
  END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
  write_status "blocked" "candidate gate blocked; END_BJT=$END_BJT"
  echo "END_BJT=$END_BJT" | tee -a "$LOG_DIR/run_info.txt"
  exit 0
}
run_short_train_if_enabled || true

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "queue" "all_done" "summary=$SUMMARY end=$END_BJT"
write_status "done" "END_BJT=$END_BJT SUMMARY=$SUMMARY"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "EVENTS=$EVENTS"
  echo "STATUS_JSON=$STATUS_JSON"
  echo "VALIDATOR_SUMMARY=$VALIDATOR_DIR/candidate_validation_summary.json"
  echo "GATE_JSON=$GATE_JSON"
} | tee -a "$LOG_DIR/run_info.txt"
