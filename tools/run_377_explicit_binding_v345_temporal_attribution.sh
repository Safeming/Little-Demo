#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
RUN_ID="${RUN_ID:-v345_temporal_attribution_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v345_temporal_attribution_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v345_temporal_attribution_${RUN_ID}}"
HYDRA_RUN_ROOT="${HYDRA_RUN_ROOT:-$LOG_DIR/hydra_runtime}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
WINDOW_SPECS="${WINDOW_SPECS:-c21_f160_162|[21]|[160,162,1];c23_f298_300|[23]|[298,300,1];c21_f266_267|[21]|[266,267,1];c22_f144_145|[22]|[144,145,1];c22_f213_214|[22]|[213,214,1]}"
RENDER_EXPORT_OPACITY_THRESHOLD="${RENDER_EXPORT_OPACITY_THRESHOLD:-0.06}"

COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/assets/adopted_geometry/377/v320_selected_components.csv}"
SIGNED_POINT_JSON="${SIGNED_POINT_JSON:-$ROOT/assets/adopted_geometry/377/v338_temporal_selector_grow_only_guard.json}"
V344_ROW_GUARD_JSON="${V344_ROW_GUARD_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v344_component_row_guard_probe_v344_component_row_guard_geometry_20260523/assets/v344_component_row_guard_top4.json}"

SUMMARY="$LOG_DIR/summary.tsv"
ATTRIBUTION="$LOG_DIR/attribution.tsv"
EVENTS="$LOG_DIR/events.tsv"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$CANDIDATE_CKPT" "$DATA_ROOT" \
  "$COMPONENT_CSV" "$SIGNED_POINT_JSON" \
  "$ROOT/assets/adopted_geometry/377/manifest.json" \
  "$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

if [ ! -e "$V344_ROW_GUARD_JSON" ]; then
  echo "warning: V344_ROW_GUARD_JSON missing, v344_row_guard_top4 branch will be skipped: $V344_ROW_GUARD_JSON" >&2
  V344_ROW_GUARD_JSON=""
fi

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
)

render_and_analyze() {
  local window="$1"
  local views_spec="$2"
  local frames_spec="$3"
  local variant="$4"
  local ckpt="$5"
  local render_exp="$6"
  shift 6

  log_event "render_${window}_${variant}_start" "$render_exp"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.subject=CoreView_377" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=$views_spec" \
    "dataset.test_frames.view=$frames_spec" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=true" \
    "++render_export_refine=false" \
    "++render_export_opacity_threshold=$RENDER_EXPORT_OPACITY_THRESHOLD" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/${window}_${variant}" \
    "wandb_disable=true" \
    "$@" \
    > "$LOG_DIR/render_${window}_${variant}.log" 2>&1
  log_event "render_${window}_${variant}_done" "status=0"

  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${window}_${variant}.log" 2>&1

  "$PYTHON_BIN" tools/analyze_377_boundary_residuals.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --render-support-threshold 0.025 \
    --close-kernel 5 \
    --band-width 7 \
    --search-band-width 24 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/boundary_residuals" \
    > "$LOG_DIR/boundary_residuals_${window}_${variant}.log" 2>&1

  "$PYTHON_BIN" tools/analyze_377_opacity_footprint.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --render-support-threshold 0.025 \
    --primary-opacity-threshold 0.06 \
    --opacity-thresholds 0.02,0.04,0.06,0.08,0.10 \
    --rgb-close-kernel 5 \
    --opacity-close-kernel 3 \
    --band-width 7 \
    --search-band-width 24 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/opacity_footprint" \
    > "$LOG_DIR/opacity_footprint_${window}_${variant}.log" 2>&1
  log_event "analyze_${window}_${variant}_done" "status=0"
}

run_window() {
  local window="$1"
  local views_spec="$2"
  local frames_spec="$3"
  local window_root="$EXP_ROOT/$window"

  render_and_analyze "$window" "$views_spec" "$frames_spec" baseline_no_preset "$BASE_CKPT" "$window_root/baseline_no_preset" \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_dynamic_enable=false" \
    "++pipeline.covariance_signed_point_json=" \
    "++pipeline.covariance_signed_point_screen_actuator_enable=false" \
    "++pipeline.covariance_signed_center_offset_enable=false" \
    "++model.deformer.rigid.geometry_fidelity_gate_enable=false"

  render_and_analyze "$window" "$views_spec" "$frames_spec" formal_v338_current "$CANDIDATE_CKPT" "$window_root/formal_v338_current" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard"

  render_and_analyze "$window" "$views_spec" "$frames_spec" no_signed_point_json "$CANDIDATE_CKPT" "$window_root/no_signed_point_json" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_disable_signed_point_json=true"

  render_and_analyze "$window" "$views_spec" "$frames_spec" no_point_screen_actuator "$CANDIDATE_CKPT" "$window_root/no_point_screen_actuator" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_disable_signed_point_screen_actuator=true"

  render_and_analyze "$window" "$views_spec" "$frames_spec" no_signed_dynamic "$CANDIDATE_CKPT" "$window_root/no_signed_dynamic" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_disable_signed_dynamic=true"

  render_and_analyze "$window" "$views_spec" "$frames_spec" no_geometry_fidelity "$CANDIDATE_CKPT" "$window_root/no_geometry_fidelity" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_disable_geometry_fidelity=true"

  render_and_analyze "$window" "$views_spec" "$frames_spec" drop_signed_dynamic_outer "$CANDIDATE_CKPT" "$window_root/drop_signed_dynamic_outer" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_signed_dynamic_over_drop_images='$(
      "$PYTHON_BIN" - "$views_spec" "$frames_spec" <<'PY'
import ast
import sys
views = ast.literal_eval(sys.argv[1])
start, end, step = ast.literal_eval(sys.argv[2])
print(",".join(f"c{int(v):02d}_f{int(f):06d}" for v in views for f in range(int(start), int(end) + 1, int(step))))
PY
    )'"

  render_and_analyze "$window" "$views_spec" "$frames_spec" drop_signed_dynamic_inner "$CANDIDATE_CKPT" "$window_root/drop_signed_dynamic_inner" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_signed_dynamic_under_drop_images='$(
      "$PYTHON_BIN" - "$views_spec" "$frames_spec" <<'PY'
import ast
import sys
views = ast.literal_eval(sys.argv[1])
start, end, step = ast.literal_eval(sys.argv[2])
print(",".join(f"c{int(v):02d}_f{int(f):06d}" for v in views for f in range(int(start), int(end) + 1, int(step))))
PY
    )'"

  if [ -n "$V344_ROW_GUARD_JSON" ]; then
    render_and_analyze "$window" "$views_spec" "$frames_spec" v344_row_guard_top4 "$CANDIDATE_CKPT" "$window_root/v344_row_guard_top4" \
      "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
      "++explicit_binding_adopted_signed_dynamic_component_row_guard_json=$V344_ROW_GUARD_JSON"
  fi
}

IFS=';' read -r -a WINDOWS <<< "$WINDOW_SPECS"
for item in "${WINDOWS[@]}"; do
  IFS='|' read -r window views frames <<< "$item"
  run_window "$window" "$views" "$frames"
done

"$PYTHON_BIN" tools/analyze_377_stageB_v345_attribution.py \
  --exp-root "$EXP_ROOT" \
  --component-csv "$COMPONENT_CSV" \
  --signed-point-json "$SIGNED_POINT_JSON" \
  --summary-out "$SUMMARY" \
  --attribution-out "$ATTRIBUTION"

log_event "summary_done" "$SUMMARY"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "ATTRIBUTION=$ATTRIBUTION"
