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
RUN_ID="${RUN_ID:-stageB_v269_visual_hull_renderloop_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v269_visual_hull_renderloop_${RUN_ID}}"
mkdir -p "$LOG_DIR"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_shoes_preserve_control_stageB_compact_v233_skincloth_20260512_161652_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt135710.pth}"
BASELINE_JSON="${BASELINE_JSON:-$ROOT/exp/stageB/logs/377_stageB_v263c_footprint_smallradius_manual_20260516_2030_bjt/v233d_precise_testview_baseline.json}"
BASE_RENDER_EXP="${BASE_RENDER_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_stageC_softer_stageC_v235_v236_20260513_114036_bjt_render_compact_final}"

TRAIN_NAME="${TRAIN_NAME:-v269a_visual_hull_renderloop}"
APPEND_ITER="${APPEND_ITER:-135711}"
STAGEB_ITERATIONS="${STAGEB_ITERATIONS:-100}"
FINAL_ITER="$((APPEND_ITER + STAGEB_ITERATIONS))"

SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
STATUS_JSON="$LOG_DIR/status.json"
VARIANT_SCORE_JSON="$LOG_DIR/v269_variant_scores.json"
BEST_INFO_JSON="$LOG_DIR/v269_best_variant.json"
FINAL_GATE_JSON="$LOG_DIR/v269_final_strong_gate.json"
SEARCH_CANDIDATE_DIR="$LOG_DIR/v269_visual_hull_candidates"
SEARCH_MAX_CANDIDATES="${SEARCH_MAX_CANDIDATES:-16}"
VISUAL_HULL_CANDIDATES_CSV="$SEARCH_CANDIDATE_DIR/visual_hull_candidates.csv"
VISUAL_HULL_CANDIDATE_SUMMARY="$SEARCH_CANDIDATE_DIR/visual_hull_candidate_summary.json"
SEARCH_MIN_INNER_GAIN="${SEARCH_MIN_INNER_GAIN:-0.5}"
SEARCH_OUTER_TOLERANCE="${SEARCH_OUTER_TOLERANCE:-0.0}"
SEARCH_RGB_TOLERANCE="${SEARCH_RGB_TOLERANCE:-0.0}"
SEARCH_EDGE_TOLERANCE="${SEARCH_EDGE_TOLERANCE:-0.0}"

WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-18000}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-65}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"

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
  for required in "$PYTHON_BIN" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$BASELINE_JSON" "$BASE_RENDER_EXP/test-view/renders"; do
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

run_append_variant() {
  local variant="$1"
  local max_candidates="$2"
  local patch_pattern="$3"
  local patch_radius="$4"
  local opacity_factor="$5"
  local opacity_floor="$6"
  local opacity_ceiling="$7"
  local scale_factor="$8"
  local scale_max="$9"
  local append_exp="$ROOT/exp/stageB/377_hulk_light_${TRAIN_NAME}_${variant}_append_${RUN_ID}"
  local render_exp="${append_exp}_render_v269_ckpt${APPEND_ITER}"
  local candidate_csv="$VISUAL_HULL_CANDIDATES_CSV"
  local candidate_summary="$VISUAL_HULL_CANDIDATE_SUMMARY"
  wait_for_gpu "$variant"
  if [ ! -s "$candidate_csv" ]; then
    log_event "$variant" "candidate_search_missing" "$candidate_csv"
    summary_row "$variant" "candidate_search" "" "" "$candidate_csv" "failed" "missing visual hull candidates"
    return 2
  fi
  summary_row "$variant" "candidate_search" "" "" "$candidate_csv" "ok" "$candidate_summary"
  log_event "$variant" "append_start" "max=${max_candidates} pattern=${patch_pattern} r=${patch_radius} op=${opacity_factor}/${opacity_floor}/${opacity_ceiling} sc=${scale_factor}/${scale_max} candidates=$candidate_csv"
  common_env "$PYTHON_BIN" "$ROOT/tools/append_377_stageB_v266_micro_patch_support.py" \
    --config-path "$BASE_EXP/.hydra/config.yaml" \
    --load-ckpt "$BASE_CKPT" \
    --candidates-csv "$candidate_csv" \
    --out-dir "$append_exp" \
    --dataset-root "$DATA_ROOT" \
    --parser-root "$PARSER_ROOT" \
    --compact-mapping "$COMPACT_MAPPING" \
    --max-candidates "$max_candidates" \
    --checkpoint-iteration "$APPEND_ITER" \
    --parent-screen-radius 34.0 \
    --patch-pattern "$patch_pattern" \
    --patch-radius-world "$patch_radius" \
    --child-opacity-factor "$opacity_factor" \
    --child-opacity-floor "$opacity_floor" \
    --child-opacity-ceiling "$opacity_ceiling" \
    --child-scale-factor "$scale_factor" \
    --child-scale-max "$scale_max" \
    > "$LOG_DIR/${variant}_append.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "$variant" "append_failed" "status=$status"
    summary_row "$variant" "append" "$append_exp" "" "$append_exp/ckpt${APPEND_ITER}.pth" "failed" "status=$status"
    return "$status"
  fi
  log_event "$variant" "append_done" "$append_exp/ckpt${APPEND_ITER}.pth"
  summary_row "$variant" "append" "$append_exp" "" "$append_exp/ckpt${APPEND_ITER}.pth" "ok" "$append_exp/v266_micro_patch_append_summary.json"
  render_ckpt "$append_exp" "$append_exp/ckpt${APPEND_ITER}.pth" "$render_exp" "${variant}_render" || return $?
  "$PYTHON_BIN" - "$VARIANT_SCORE_JSON.tmp" "$variant" "$append_exp" "$render_exp" "$BASELINE_JSON" "$render_exp/diagnostics/${RENDER_SPLIT_DIR}_boundary_residuals/boundary_residual_summary.json" "$render_exp/diagnostics/${RENDER_SPLIT_DIR}_contours/contour_summary.json" "$candidate_csv" "$candidate_summary" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
candidate_csv_p = Path(sys.argv[8])
candidate_summary_p = Path(sys.argv[9])
variant = sys.argv[2]
append_exp = sys.argv[3]
render_exp = sys.argv[4]
base = json.loads(Path(sys.argv[5]).read_text())
res = json.loads(Path(sys.argv[6]).read_text())
con = json.loads(Path(sys.argv[7]).read_text())
inner = float(res["mean_inner_missing_pixels"])
outer = float(res["mean_outer_leak_pixels"])
fg = float(con["mean_fg_l1"])
bd = float(con["mean_boundary_l1"])
edge = float(con["mean_edge_symmetric_dist_px"])
inner_gain = float(base["mean_inner_missing_pixels"]) - inner
outer_delta = outer - float(base["mean_outer_leak_pixels"])
fg_delta = fg - float(base["mean_fg_l1"])
bd_delta = bd - float(base["mean_boundary_l1"])
edge_delta = edge - float(base["mean_edge_symmetric_dist_px"])
score = (
    inner_gain
    - 5.0 * max(outer_delta, 0.0)
    - 900.0 * max(fg_delta, 0.0)
    - 900.0 * max(bd_delta, 0.0)
    - 4.0 * max(edge_delta, 0.0)
)
record = {
    "variant": variant,
    "append_exp": append_exp,
    "render_exp": render_exp,
    "ckpt": str(Path(append_exp) / "ckpt135711.pth"),
    "candidate_csv": str(candidate_csv_p),
    "candidate_summary": str(candidate_summary_p),
    "score": score,
    "inner_gain": inner_gain,
    "outer_delta": outer_delta,
    "fg_delta": fg_delta,
    "boundary_delta": bd_delta,
    "edge_delta": edge_delta,
    "metrics": {
        "mean_inner_missing_pixels": inner,
        "mean_outer_leak_pixels": outer,
        "mean_fg_l1": fg,
        "mean_boundary_l1": bd,
        "mean_edge_symmetric_dist_px": edge,
    },
}
records = []
if out.exists():
    records = json.loads(out.read_text())
records.append(record)
out.write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2), flush=True)
PY
  cp "$VARIANT_SCORE_JSON.tmp" "$VARIANT_SCORE_JSON"
  return 0
}

select_best_variant() {
  "$PYTHON_BIN" - "$VARIANT_SCORE_JSON" "$BEST_INFO_JSON" "$SEARCH_MIN_INNER_GAIN" "$SEARCH_OUTER_TOLERANCE" "$SEARCH_RGB_TOLERANCE" "$SEARCH_EDGE_TOLERANCE" <<'PY'
import json, sys
from pathlib import Path
records = json.loads(Path(sys.argv[1]).read_text())
if not records:
    raise SystemExit("no variant records")
min_inner_gain = float(sys.argv[3])
outer_tol = float(sys.argv[4])
rgb_tol = float(sys.argv[5])
edge_tol = float(sys.argv[6])
records.sort(key=lambda r: (r["score"], r["inner_gain"], -r["outer_delta"]), reverse=True)
best = records[0]
eligible = [
    r for r in records
    if r["inner_gain"] >= min_inner_gain
    and r["outer_delta"] <= outer_tol
    and r["fg_delta"] <= rgb_tol
    and r["boundary_delta"] <= rgb_tol
    and r["edge_delta"] <= edge_tol
]
payload = {
    "status": "ok" if eligible else "blocked",
    "best": eligible[0] if eligible else best,
    "best_raw": best,
    "eligible_count": len(eligible),
    "criteria": {
        "min_inner_gain": min_inner_gain,
        "outer_tolerance": outer_tol,
        "rgb_tolerance": rgb_tol,
        "edge_tolerance": edge_tol,
    },
    "records": records,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(best, indent=2), flush=True)
if not eligible:
    raise SystemExit(2)
PY
}

json_get() {
  local json_path="$1"
  local expr="$2"
  "$PYTHON_BIN" - "$json_path" "$expr" <<'PY'
import json, sys
data=json.loads(open(sys.argv[1]).read())
cur=data
for part in sys.argv[2].split('.'):
    cur=cur[part]
print(cur)
PY
}

run_support_only_train() {
  local best_append_exp="$1"
  local best_variant="$2"
  local train_exp="$ROOT/exp/stageB/377_hulk_light_${TRAIN_NAME}_${best_variant}_train_${RUN_ID}"
  local final_render_exp="${train_exp}_render_v269_ckpt${FINAL_ITER}"
  mkdir -p "$train_exp"
  wait_for_gpu "${TRAIN_NAME}_${best_variant}"
  log_event "${TRAIN_NAME}_${best_variant}" "train_start" "ckpt=$best_append_exp/ckpt${APPEND_ITER}.pth iterations=$STAGEB_ITERATIONS final_iter=$FINAL_ITER"
  common_env "$PYTHON_BIN" "$ROOT/train.py" \
    --config-path "$best_append_exp/.hydra" \
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
    "start_checkpoint=$best_append_exp/ckpt${APPEND_ITER}.pth" \
    "exp_dir=$train_exp" \
    "seed=${SEED:--1}" \
    "hydra.run.dir=$LOG_DIR/hydra_${TRAIN_NAME}_${best_variant}_train" \
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
    > "$LOG_DIR/${TRAIN_NAME}_${best_variant}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "${TRAIN_NAME}_${best_variant}" "train_failed" "status=$status"
    summary_row "${TRAIN_NAME}_${best_variant}" "train" "$train_exp" "" "$train_exp/ckpt${FINAL_ITER}.pth" "failed" "status=$status"
    return "$status"
  fi
  summary_row "${TRAIN_NAME}_${best_variant}" "train" "$train_exp" "" "$train_exp/ckpt${FINAL_ITER}.pth" "ok" "iterations=$STAGEB_ITERATIONS"
  log_event "${TRAIN_NAME}_${best_variant}" "train_done" "$train_exp/ckpt${FINAL_ITER}.pth"
  render_ckpt "$train_exp" "$train_exp/ckpt${FINAL_ITER}.pth" "$final_render_exp" "v269_final_render" || true
  "$PYTHON_BIN" "$ROOT/tools/check_377_stageB_v261_do_no_harm.py" \
    --candidate-summary "$VISUAL_HULL_CANDIDATE_SUMMARY" \
    --require-candidate-ok \
    --residual-summary "$final_render_exp/diagnostics/${RENDER_SPLIT_DIR}_boundary_residuals/boundary_residual_summary.json" \
    --contour-summary "$final_render_exp/diagnostics/${RENDER_SPLIT_DIR}_contours/contour_summary.json" \
    --baseline-json "$BASELINE_JSON" \
    --inner-epsilon "${INNER_EPSILON:-10.0}" \
    --outer-tolerance "${OUTER_TOLERANCE:-0.0}" \
    --rgb-tolerance "${RGB_TOLERANCE:-0.0000}" \
    --edge-tolerance "${EDGE_TOLERANCE:-0.0000}" \
    --out-json "$FINAL_GATE_JSON" \
    > "$LOG_DIR/v269_final_strong_gate.log" 2>&1 || true
  summary_row "v269_final_strong_gate" "gate" "" "$final_render_exp" "" "$([ -s "$FINAL_GATE_JSON" ] && "$PYTHON_BIN" - "$FINAL_GATE_JSON" <<'PY' || echo unknown
import json, sys
print(json.load(open(sys.argv[1])).get("status", "unknown"))
PY
)" "$FINAL_GATE_JSON"
  echo "$train_exp" > "$LOG_DIR/final_train_exp.txt"
  echo "$final_render_exp" > "$LOG_DIR/final_render_exp.txt"
}

{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "GPU=$GPU"
  echo "BASE_CKPT=$BASE_CKPT"
  echo "BASE_RENDER_EXP=$BASE_RENDER_EXP"
  echo "LOG_DIR=$LOG_DIR"
  echo "VARIANTS=vh_cross5_small,vh_cross5_mid,vh_diamond9_small"
  echo "SEARCH_MAX_CANDIDATES=$SEARCH_MAX_CANDIDATES"
  echo "SEARCH_MIN_INNER_GAIN=$SEARCH_MIN_INNER_GAIN"
  echo "STAGEB_ITERATIONS=$STAGEB_ITERATIONS FINAL_ITER=$FINAL_ITER"
} | tee "$LOG_DIR/run_info.txt"

write_status "starting" "preflight"
check_required || exit $?
rm -f "$VARIANT_SCORE_JSON" "$VARIANT_SCORE_JSON.tmp"

mkdir -p "$SEARCH_CANDIDATE_DIR"
wait_for_gpu "v269_visual_hull_candidates"
log_event "v269_visual_hull_candidates" "candidate_search_start" "base_render=$BASE_RENDER_EXP max_out=48"
common_env "$PYTHON_BIN" "$ROOT/tools/make_377_stageB_v269_visual_hull_candidates.py" \
  --config-path "$BASE_EXP/.hydra/config.yaml" \
  --load-ckpt "$BASE_CKPT" \
  --base-render-exp "$BASE_RENDER_EXP" \
  --out-csv "$VISUAL_HULL_CANDIDATES_CSV" \
  --out-summary "$VISUAL_HULL_CANDIDATE_SUMMARY" \
  --dataset-root "$DATA_ROOT" \
  --parser-root "$PARSER_ROOT" \
  --compact-mapping "$COMPACT_MAPPING" \
  --eval-views "21,22,23" \
  --frames "0,60,120,180,240,300,360,420,480,540" \
  --depth-samples "${VISUAL_HULL_DEPTH_SAMPLES:-96}" \
  --max-output-candidates 48 \
  > "$LOG_DIR/v269_visual_hull_candidates.log" 2>&1
candidate_status=$?
if [ "$candidate_status" -ne 0 ]; then
  log_event "v269_visual_hull_candidates" "candidate_search_failed" "status=$candidate_status"
  summary_row "v269_visual_hull_candidates" "candidate_search" "" "" "$VISUAL_HULL_CANDIDATES_CSV" "failed" "$VISUAL_HULL_CANDIDATE_SUMMARY"
  exit "$candidate_status"
fi
log_event "v269_visual_hull_candidates" "candidate_search_done" "$VISUAL_HULL_CANDIDATE_SUMMARY"
summary_row "v269_visual_hull_candidates" "candidate_search" "" "" "$VISUAL_HULL_CANDIDATES_CSV" "ok" "$VISUAL_HULL_CANDIDATE_SUMMARY"

run_append_variant "vh_cross5_small_top16" "$SEARCH_MAX_CANDIDATES" cross5 0.0018 0.50 0.016 0.13 0.24 0.0045 || true
run_append_variant "vh_cross5_mid_top16" "$SEARCH_MAX_CANDIDATES" cross5 0.0025 0.50 0.016 0.13 0.26 0.0050 || true
run_append_variant "vh_diamond9_small_top16" "$SEARCH_MAX_CANDIDATES" diamond9 0.0016 0.45 0.014 0.12 0.22 0.0042 || true

select_best_variant || exit $?
BEST_VARIANT="$(json_get "$BEST_INFO_JSON" best.variant)"
BEST_APPEND_EXP="$(json_get "$BEST_INFO_JSON" best.append_exp)"
log_event "v269_select" "best" "variant=$BEST_VARIANT append_exp=$BEST_APPEND_EXP"
summary_row "v269_select" "select" "$BEST_APPEND_EXP" "$(json_get "$BEST_INFO_JSON" best.render_exp)" "$(json_get "$BEST_INFO_JSON" best.ckpt)" "ok" "$BEST_INFO_JSON"

run_support_only_train "$BEST_APPEND_EXP" "$BEST_VARIANT" || exit $?

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "queue" "all_done" "summary=$SUMMARY end=$END_BJT"
write_status "done" "END_BJT=$END_BJT SUMMARY=$SUMMARY"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "EVENTS=$EVENTS"
  echo "STATUS_JSON=$STATUS_JSON"
  echo "VARIANT_SCORE_JSON=$VARIANT_SCORE_JSON"
  echo "BEST_INFO_JSON=$BEST_INFO_JSON"
  echo "FINAL_GATE_JSON=$FINAL_GATE_JSON"
  echo "FINAL_TRAIN_EXP=$(cat "$LOG_DIR/final_train_exp.txt" 2>/dev/null || true)"
  echo "FINAL_RENDER_EXP=$(cat "$LOG_DIR/final_render_exp.txt" 2>/dev/null || true)"
} | tee -a "$LOG_DIR/run_info.txt"
