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
RUN_ID="${RUN_ID:-stageB_v266_micro_patch_support_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v266_micro_patch_support_${RUN_ID}}"
mkdir -p "$LOG_DIR"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_shoes_preserve_control_stageB_compact_v233_skincloth_20260512_161652_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt135710.pth}"
BASELINE_JSON="${BASELINE_JSON:-$ROOT/exp/stageB/logs/377_stageB_v263c_footprint_smallradius_manual_20260516_2030_bjt/v233d_precise_testview_baseline.json}"
SOURCE_CANDIDATES_CSV="${SOURCE_CANDIDATES_CSV:-$ROOT/exp/stageB/logs/377_stageB_v264b_actual_radii_tight_validator_20260516_2130_bjt/v264b_actual_radii_validator/actual_radii_accepted_candidates.csv}"
SOURCE_CANDIDATE_SUMMARY="${SOURCE_CANDIDATE_SUMMARY:-$ROOT/exp/stageB/logs/377_stageB_v264b_actual_radii_tight_validator_20260516_2130_bjt/v264b_actual_radii_validator/actual_radii_candidate_summary.json}"

TRAIN_NAME="${TRAIN_NAME:-v266a_micro_patch_support}"
BASE_ITER="${BASE_ITER:-135710}"
APPEND_ITER="${APPEND_ITER:-135711}"
STAGEB_ITERATIONS="${STAGEB_ITERATIONS:-100}"
FINAL_ITER="$((APPEND_ITER + STAGEB_ITERATIONS))"

APPEND_EXP="$ROOT/exp/stageB/377_hulk_light_${TRAIN_NAME}_append_${RUN_ID}"
APPEND_RENDER_EXP="${APPEND_EXP}_render_v266_ckpt${APPEND_ITER}"
TRAIN_EXP="$ROOT/exp/stageB/377_hulk_light_${TRAIN_NAME}_${RUN_ID}"
FINAL_RENDER_EXP="${TRAIN_EXP}_render_v266_ckpt${FINAL_ITER}"

SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
STATUS_JSON="$LOG_DIR/status.json"
APPEND_GATE_JSON="$LOG_DIR/v266_append_strong_gate.json"
FINAL_GATE_JSON="$LOG_DIR/v266_final_strong_gate.json"

WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-18000}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-65}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"

MAX_CANDIDATES="${MAX_CANDIDATES:-16}"
PATCH_PATTERN="${PATCH_PATTERN:-cross5}"
PATCH_RADIUS_WORLD="${PATCH_RADIUS_WORLD:-0.0035}"
PARENT_SCREEN_RADIUS="${PARENT_SCREEN_RADIUS:-34.0}"
CHILD_OPACITY_FACTOR="${CHILD_OPACITY_FACTOR:-0.55}"
CHILD_OPACITY_FLOOR="${CHILD_OPACITY_FLOOR:-0.018}"
CHILD_OPACITY_CEILING="${CHILD_OPACITY_CEILING:-0.16}"
CHILD_SCALE_FACTOR="${CHILD_SCALE_FACTOR:-0.30}"
CHILD_SCALE_MAX="${CHILD_SCALE_MAX:-0.0060}"

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
  for required in "$PYTHON_BIN" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$BASELINE_JSON" "$SOURCE_CANDIDATES_CSV" "$SOURCE_CANDIDATE_SUMMARY"; do
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

run_append() {
  wait_for_gpu "v266_micro_patch_append"
  log_event "v266_micro_patch_append" "start" "candidates=$SOURCE_CANDIDATES_CSV max=$MAX_CANDIDATES pattern=$PATCH_PATTERN"
  common_env "$PYTHON_BIN" "$ROOT/tools/append_377_stageB_v266_micro_patch_support.py" \
    --config-path "$BASE_EXP/.hydra/config.yaml" \
    --load-ckpt "$BASE_CKPT" \
    --candidates-csv "$SOURCE_CANDIDATES_CSV" \
    --out-dir "$APPEND_EXP" \
    --dataset-root "$DATA_ROOT" \
    --parser-root "$PARSER_ROOT" \
    --compact-mapping "$COMPACT_MAPPING" \
    --max-candidates "$MAX_CANDIDATES" \
    --checkpoint-iteration "$APPEND_ITER" \
    --parent-screen-radius "$PARENT_SCREEN_RADIUS" \
    --patch-pattern "$PATCH_PATTERN" \
    --patch-radius-world "$PATCH_RADIUS_WORLD" \
    --child-opacity-factor "$CHILD_OPACITY_FACTOR" \
    --child-opacity-floor "$CHILD_OPACITY_FLOOR" \
    --child-opacity-ceiling "$CHILD_OPACITY_CEILING" \
    --child-scale-factor "$CHILD_SCALE_FACTOR" \
    --child-scale-max "$CHILD_SCALE_MAX" \
    > "$LOG_DIR/v266_micro_patch_append.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "v266_micro_patch_append" "failed" "status=$status"
    summary_row "v266_micro_patch_append" "append" "$APPEND_EXP" "" "$APPEND_EXP/ckpt${APPEND_ITER}.pth" "failed" "status=$status"
    return "$status"
  fi
  summary_row "v266_micro_patch_append" "append" "$APPEND_EXP" "" "$APPEND_EXP/ckpt${APPEND_ITER}.pth" "ok" "$APPEND_EXP/v266_micro_patch_append_summary.json"
  log_event "v266_micro_patch_append" "done" "$APPEND_EXP/ckpt${APPEND_ITER}.pth"
}

render_ckpt() {
  local config_exp="$1"
  local ckpt="$2"
  local render_exp="$3"
  local name="$4"
  wait_for_gpu "$name"
  log_event "$name" "render_start" "$ckpt"
  common_env "$PYTHON_BIN" "$ROOT/render.py" \
    --config-path "$config_exp/.hydra" \
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
    "hydra.run.dir=$LOG_DIR/hydra_${name}" \
    "wandb_disable=true" \
    > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "$name" "render_failed" "status=$status"
    summary_row "$name" "render" "$config_exp" "$render_exp" "$ckpt" "failed" "status=$status"
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
    > "$LOG_DIR/${name}_residuals.log" 2>&1
  local residual_status=$?
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_render_contours.py" \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir "$RENDER_SPLIT_DIR" \
    --band-width 7 \
    --out-dir "$render_exp/diagnostics/${RENDER_SPLIT_DIR}_contours" \
    > "$LOG_DIR/${name}_contours.log" 2>&1
  local contour_status=$?
  if [ "$residual_status" -ne 0 ] || [ "$contour_status" -ne 0 ]; then
    log_event "$name" "metrics_failed" "residual=$residual_status contour=$contour_status"
    summary_row "$name" "render" "$config_exp" "$render_exp" "$ckpt" "failed" "metrics residual=$residual_status contour=$contour_status"
    return 2
  fi
  log_event "$name" "render_done" "$render_exp"
  summary_row "$name" "render" "$config_exp" "$render_exp" "$ckpt" "ok" "$render_exp"
}

run_gate() {
  local name="$1"
  local render_exp="$2"
  local out_json="$3"
  log_event "$name" "gate_start" "$render_exp"
  "$PYTHON_BIN" "$ROOT/tools/check_377_stageB_v261_do_no_harm.py" \
    --candidate-summary "$SOURCE_CANDIDATE_SUMMARY" \
    --require-candidate-ok \
    --residual-summary "$render_exp/diagnostics/${RENDER_SPLIT_DIR}_boundary_residuals/boundary_residual_summary.json" \
    --contour-summary "$render_exp/diagnostics/${RENDER_SPLIT_DIR}_contours/contour_summary.json" \
    --baseline-json "$BASELINE_JSON" \
    --inner-epsilon "${INNER_EPSILON:-10.0}" \
    --outer-tolerance "${OUTER_TOLERANCE:-0.0}" \
    --rgb-tolerance "${RGB_TOLERANCE:-0.0000}" \
    --edge-tolerance "${EDGE_TOLERANCE:-0.0000}" \
    --out-json "$out_json" \
    > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "$name" "blocked" "$out_json"
    summary_row "$name" "gate" "" "$render_exp" "" "blocked" "$out_json"
    return "$status"
  fi
  summary_row "$name" "gate" "" "$render_exp" "" "ok" "$out_json"
  log_event "$name" "ok" "$out_json"
}

run_support_only_train() {
  mkdir -p "$TRAIN_EXP"
  wait_for_gpu "$TRAIN_NAME"
  log_event "$TRAIN_NAME" "train_start" "ckpt=$APPEND_EXP/ckpt${APPEND_ITER}.pth iterations=$STAGEB_ITERATIONS final_iter=$FINAL_ITER"
  common_env "$PYTHON_BIN" "$ROOT/train.py" \
    --config-path "$APPEND_EXP/.hydra" \
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
    "start_checkpoint=$APPEND_EXP/ckpt${APPEND_ITER}.pth" \
    "exp_dir=$TRAIN_EXP" \
    "seed=${SEED:--1}" \
    "hydra.run.dir=$LOG_DIR/hydra_${TRAIN_NAME}_train" \
    "wandb_disable=true" \
    "++resume.allow_partial_converter_load=false" \
    "++resume.restore_gaussian_optimizer_state=false" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.use_checkpoint_iteration_as_offset=true" \
    "++resume.require_no_densify_on_resume=true" \
    "pipeline.pose_noise=0.0" \
    "model.gaussian.delay=0" \
    "++model.pose_correction.train_root_orient=false" \
    "++model.pose_correction.train_pose_body=false" \
    "++model.pose_correction.train_pose_hand=false" \
    "++model.pose_correction.train_trans=false" \
    "++model.pose_correction.train_betas=false" \
    "opt.iterations=$STAGEB_ITERATIONS" \
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
    "++opt.boundary_support_only_grad_mask_enable=true" \
    "++opt.boundary_support_only_grad_mask_verbose=false" \
    "++opt.boundary_aware_enable=false" \
    "++opt.boundary_component_support_enable=false" \
    "++model.gaussian.boundary_component_support_enable=false" \
    "++model.gaussian.boundary_support_projector_enable=false" \
    "++model.gaussian.binding_densify_disable_clone=true" \
    "++model.gaussian.binding_densify_disable_split=true" \
    "opt.position_lr_init=0.00000042" \
    "opt.position_lr_final=0.00000012" \
    "opt.opacity_lr=0.000014" \
    "opt.scaling_lr=0.00000055" \
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "opt.lambda_l1=0.45" \
    "opt.lambda_l1_fg=0.75" \
    "opt.lambda_l1_boundary=0.28" \
    "opt.lambda_perceptual=0.002" \
    "opt.lambda_mask=0.0025" \
    "++opt.lambda_mask_boundary=0.0035" \
    "++opt.lambda_mask_boundary_hard=0.0025" \
    "++opt.lambda_silhouette_inner=0.0045" \
    "++opt.lambda_silhouette_outer=0.0012" \
    "++opt.lambda_silhouette_outer_shell=0.0018" \
    "++opt.silhouette_inner_ring_width=7" \
    "++opt.silhouette_outer_ring_width=5" \
    "++opt.silhouette_outer_shell_start_width=1" \
    "++opt.silhouette_outer_shell_end_width=13" \
    "++opt.lambda_detail_face=0.0" \
    "++opt.lambda_detail_shoulder_arm=0.0" \
    "++opt.lambda_detail_waist=0.0" \
    "opt.lambda_skinning=0.0" \
    "opt.lambda_aiap_xyz=0.0" \
    "opt.lambda_aiap_cov=0.0" \
    "test_interval=1000" \
    "test_iterations=[$STAGEB_ITERATIONS]" \
    "save_iterations=[$STAGEB_ITERATIONS]" \
    "checkpoint_iterations=[$STAGEB_ITERATIONS]" \
    "++validation_image_log_limit=0" \
    "opt.percent_dense=0.0" \
    "opt.densify_from_iter=0" \
    "opt.densify_until_iter=0" \
    "opt.densification_interval=1000000" \
    "opt.opacity_threshold=0.000001" \
    "opt.opacity_reset_interval=1000000" \
    "opt.grad_clip=0.0015" \
    > "$LOG_DIR/${TRAIN_NAME}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "$TRAIN_NAME" "train_failed" "status=$status"
    summary_row "$TRAIN_NAME" "train" "$TRAIN_EXP" "" "$TRAIN_EXP/ckpt${FINAL_ITER}.pth" "failed" "status=$status"
    return "$status"
  fi
  summary_row "$TRAIN_NAME" "train" "$TRAIN_EXP" "" "$TRAIN_EXP/ckpt${FINAL_ITER}.pth" "ok" "iterations=$STAGEB_ITERATIONS"
  log_event "$TRAIN_NAME" "train_done" "$TRAIN_EXP/ckpt${FINAL_ITER}.pth"
}

{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "GPU=$GPU"
  echo "BASE_CKPT=$BASE_CKPT"
  echo "APPEND_EXP=$APPEND_EXP"
  echo "TRAIN_EXP=$TRAIN_EXP"
  echo "FINAL_ITER=$FINAL_ITER"
  echo "LOG_DIR=$LOG_DIR"
  echo "MAX_CANDIDATES=$MAX_CANDIDATES PATCH_PATTERN=$PATCH_PATTERN PATCH_RADIUS_WORLD=$PATCH_RADIUS_WORLD"
} | tee "$LOG_DIR/run_info.txt"

write_status "starting" "preflight"
check_required || exit $?
run_append || exit $?
render_ckpt "$APPEND_EXP" "$APPEND_EXP/ckpt${APPEND_ITER}.pth" "$APPEND_RENDER_EXP" "v266_append_render" || true
run_gate "v266_append_strong_gate" "$APPEND_RENDER_EXP" "$APPEND_GATE_JSON" || true
run_support_only_train || exit $?
render_ckpt "$TRAIN_EXP" "$TRAIN_EXP/ckpt${FINAL_ITER}.pth" "$FINAL_RENDER_EXP" "v266_final_render" || true
run_gate "v266_final_strong_gate" "$FINAL_RENDER_EXP" "$FINAL_GATE_JSON" || true

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "queue" "all_done" "summary=$SUMMARY end=$END_BJT"
write_status "done" "END_BJT=$END_BJT SUMMARY=$SUMMARY"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "EVENTS=$EVENTS"
  echo "STATUS_JSON=$STATUS_JSON"
  echo "APPEND_GATE_JSON=$APPEND_GATE_JSON"
  echo "FINAL_GATE_JSON=$FINAL_GATE_JSON"
  echo "APPEND_RENDER_EXP=$APPEND_RENDER_EXP"
  echo "FINAL_RENDER_EXP=$FINAL_RENDER_EXP"
} | tee -a "$LOG_DIR/run_info.txt"
