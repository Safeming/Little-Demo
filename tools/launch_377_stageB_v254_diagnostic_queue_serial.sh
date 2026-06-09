#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-stageB_v254_diagnostic_queue_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v254_diagnostic_queue_${RUN_ID}}"
mkdir -p "$LOG_DIR"

NOHUP_LOG="$LOG_DIR/nohup.log"
QUEUE_SCRIPT="$LOG_DIR/run_v254_queue.sh"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"

BASE_V250_EXP="$ROOT/exp/stageB/377_hulk_light_v250b_gapaware_component_support_9h_stageB_v250_gapaware_component_support_9h_20260515_002233_bjt"
BASE_CKPT_DEFAULT="$BASE_V250_EXP/ckpt144950.pth"
BASE_ITER_DEFAULT="144950"
V253_EXP_DEFAULT="$ROOT/exp/stageB/377_hulk_light_v253b_support_activation_probe_stageB_v253_support_activation_probe_20260515_131317_bjt"
V253_CKPT_DEFAULT="$V253_EXP_DEFAULT/ckpt145190.pth"

cat > "$QUEUE_SCRIPT" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

ROOT="__ROOT__"
cd "$ROOT"

RUN_ID="__RUN_ID__"
GPU="__GPU__"
LOG_DIR="__LOG_DIR__"
PYTHON_BIN="__PYTHON_BIN__"
BASE_CKPT_DEFAULT="__BASE_CKPT_DEFAULT__"
BASE_ITER_DEFAULT="__BASE_ITER_DEFAULT__"
V253_CKPT_DEFAULT="__V253_CKPT_DEFAULT__"
DATA_ROOT="$ROOT/data/ZJUMoCap"
PARSER_ROOT="$ROOT/data/parsers_from_hulk_multiview"
COMPACT_MAPPING="$ROOT/configs/semantic/hulk_cihp_compact_6.json"
BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin,boundary_support]"
MONTAGE_PANELS="gt render layer region body_prob cloth_prob compact_semantic boundary_support thin semantic"
mkdir -p "$LOG_DIR"

STATUS_JSON="$LOG_DIR/status.json"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
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

copy_summary_rows() {
  local src="$1"
  if [ -f "$src" ]; then
    tail -n +2 "$src" >> "$SUMMARY" || true
  fi
}

gpu_stats() {
  nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
}

wait_for_gpu() {
  local name="$1"
  local used util
  while true; do
    read -r used util < <(gpu_stats)
    used="${used:-0}"
    util="${util:-0}"
    if [ "$used" -le 18000 ] && [ "$util" -le 65 ]; then
      log_event "$name" "gpu_ready" "used=${used}MiB util=${util}%"
      return 0
    fi
    log_event "$name" "gpu_wait" "used=${used}MiB util=${util}%"
    sleep 60
  done
}

common_env() {
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
    "$@"
}

run_render_only() {
  local name="$1"
  local ckpt="$2"
  local views="$3"
  local frames="$4"
  local render_exp="$ROOT/exp/stageB/377_hulk_light_v254a_${name}_${RUN_ID}"
  local ckpt_dir
  ckpt_dir="$(dirname "$ckpt")"
  local local_log="$LOG_DIR/${name}"
  mkdir -p "$local_log"
  wait_for_gpu "$name"
  log_event "$name" "render_start" "views=$views ckpt=$ckpt"
  common_env "$PYTHON_BIN" "$ROOT/render.py" \
    --config-path "$ckpt_dir/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=$views" \
    "dataset.test_frames.view=$frames" \
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
    "hydra.run.dir=$local_log/hydra_${name}_render" \
    "wandb_disable=true" \
    > "$local_log/render.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    summary_row "$name" "render" "" "$render_exp" "$ckpt" "failed" "status=$status"
    log_event "$name" "render_failed" "status=$status"
    return 0
  fi
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_render_contours.py" \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/test-view_contours" \
    > "$local_log/contours.log" 2>&1 || true
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_boundary_residuals.py" \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --search-band-width 24 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/test-view_boundary_residuals" \
    > "$local_log/residuals.log" 2>&1 || true
  "$PYTHON_BIN" "$ROOT/tools/make_binding_paper_montage.py" \
    --exp-dir "$render_exp" \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --split test-view \
    --panels $MONTAGE_PANELS \
    --limit 8 \
    --output-dir "$render_exp/test-view/paper_montages_selected" \
    > "$local_log/montage.log" 2>&1 || true
  summary_row "$name" "render" "" "$render_exp" "$ckpt" "ok" "views=$views"
  log_event "$name" "render_done" "$render_exp"
}

run_probe() {
  local name="$1"
  local train_name="$2"
  local iterations="$3"
  local ckpt_steps="$4"
  local render_steps="$5"
  local densify_from="$6"
  local densify_until="$7"
  local densify_interval="$8"
  local coverage="$9"
  local append_only="${10}"
  local local_run_id="${RUN_ID}_${name}"
  local local_log="$LOG_DIR/${name}"
  mkdir -p "$local_log"
  log_event "$name" "start" "iterations=$iterations ckpts=$ckpt_steps append_only=$append_only coverage=$coverage"
  env \
    EXPERIMENT_VERSION="v254" \
    EXPERIMENT_SLUG="$name" \
    TRAIN_NAME="$train_name" \
    LOG_PREFIX="377_stageB_v254_${name}" \
    RENDER_TAG="v254" \
    PURPOSE="v254 diagnostic probe: append-only/coverage test for boundary component support placement" \
    RUN_ID="$local_run_id" \
    GPU="$GPU" \
    LOG_DIR="$local_log" \
    WAIT_FOR_FREE_GPU=1 \
    DO_RENDER=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    BASE_CKPT="$BASE_CKPT_DEFAULT" \
    BASE_ITER="$BASE_ITER_DEFAULT" \
    STAGEB_ITERATIONS="$iterations" \
    STAGEB_CHECKPOINT_STEPS="$ckpt_steps" \
    RENDER_CHECKPOINT_STEPS="$render_steps" \
    TEST_INTERVAL=1000 \
    DENSIFY_FROM_STEP="$densify_from" \
    DENSIFY_UNTIL_STEP="$densify_until" \
    DENSIFY_INTERVAL="$densify_interval" \
    BINDING_DENSIFY_DISABLE_CLONE="$append_only" \
    BINDING_DENSIFY_DISABLE_SPLIT="$append_only" \
    BOUNDARY_COMPONENT_SUPPORT_COVERAGE_VERBOSE="$coverage" \
    BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin,boundary_support]" \
    MONTAGE_PANELS="gt render layer region body_prob cloth_prob compact_semantic boundary_support thin semantic" \
    BOUNDARY_COMPONENT_SUPPORT_ACCUMULATE_PENDING=true \
    BOUNDARY_COMPONENT_SUPPORT_PENDING_MAX_VIEWS=4 \
    BOUNDARY_COMPONENT_SUPPORT_PENDING_MAX_POINTS=96 \
    BOUNDARY_COMPONENT_SUPPORT_PENDING_VERBOSE=true \
    BOUNDARY_COMPONENT_SUPPORT_MAX_POINTS=64 \
    BOUNDARY_COMPONENT_SUPPORT_UNIQUE_PARENT=true \
    BOUNDARY_COMPONENT_SUPPORT_RESIDUAL_THRESHOLD=0.50 \
    BOUNDARY_COMPONENT_SUPPORT_MIN_AREA=18 \
    BOUNDARY_COMPONENT_SUPPORT_MAX_COMPONENTS=12 \
    BOUNDARY_COMPONENT_SUPPORT_POINTS_PER_COMPONENT=2 \
    BOUNDARY_COMPONENT_SUPPORT_MAX_POINTS_PER_VIEW=24 \
    BOUNDARY_COMPONENT_SUPPORT_ANCHOR_SEARCH_RADIUS=24.0 \
    BOUNDARY_COMPONENT_SUPPORT_ANCHOR_FALLBACK_RADIUS=40.0 \
    BOUNDARY_COMPONENT_SUPPORT_MAX_SCREEN_GAP=38.0 \
    BOUNDARY_COMPONENT_SUPPORT_ANCHOR_BOUNDARY_MIN=0.08 \
    BOUNDARY_COMPONENT_SUPPORT_ANCHOR_OVER_MAX=0.22 \
    BOUNDARY_COMPONENT_SUPPORT_ANCHOR_OVER_REJECT_STRICT=false \
    BOUNDARY_COMPONENT_SUPPORT_ANCHOR_OVER_PENALTY_WEIGHT=0.55 \
    BOUNDARY_COMPONENT_SUPPORT_OFFSET_SCALE=0.60 \
    BOUNDARY_COMPONENT_SUPPORT_OFFSET_MIN=0.0012 \
    BOUNDARY_COMPONENT_SUPPORT_OFFSET_MAX=0.034 \
    BOUNDARY_COMPONENT_SUPPORT_SCREEN_GAP_OFFSET_SCALE=0.95 \
    BOUNDARY_COMPONENT_SUPPORT_SCREEN_GAP_RADIUS_MIN=1.5 \
    BOUNDARY_COMPONENT_SUPPORT_SCREEN_GAP_RATIO_MAX=2.5 \
    BOUNDARY_COMPONENT_SUPPORT_SCREEN_GAP_OFFSET_MAX=0.034 \
    BOUNDARY_COMPONENT_SUPPORT_CHILD_OPACITY_FACTOR=0.95 \
    BOUNDARY_COMPONENT_SUPPORT_CHILD_OPACITY_FLOOR=0.045 \
    BOUNDARY_COMPONENT_SUPPORT_CHILD_OPACITY_CEILING=0.42 \
    BOUNDARY_COMPONENT_SUPPORT_CHILD_SCALE_FACTOR=0.72 \
    BOUNDARY_SUPPORT_PROJECTOR_ENABLE=false \
    BOUNDARY_SUPPORT_PROJECTOR_VERBOSE=true \
    BOUNDARY_SUPPORT_PROJECTOR_INNER_ENABLE=false \
    BOUNDARY_SUPPORT_PROJECTOR_OUTER_ENABLE=false \
    bash "$ROOT/tools/run_377_stageB_v248_boundary_component_support_serial.sh"
  copy_summary_rows "$local_log/summary.tsv"
  log_event "$name" "done" "summary=$local_log/summary.tsv"
}

write_status "starting" "v254 diagnostic queue"
{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "GPU=$GPU"
  echo "LOG_DIR=$LOG_DIR"
  echo "BASE_CKPT=$BASE_CKPT_DEFAULT"
  echo "V253_CKPT=$V253_CKPT_DEFAULT"
  echo "SUMMARY=$SUMMARY"
  echo "EVENTS=$EVENTS"
} | tee "$LOG_DIR/run_info.txt"

run_render_only "train_view_v253_final" "$V253_CKPT_DEFAULT" "[1,2,3,4,5,6]" "[0,570,60]"
run_render_only "test_view_v253_final" "$V253_CKPT_DEFAULT" "[21,22,23]" "[0,570,60]"
run_probe "append_only_immediate" "v254b_append_only_immediate" 130 "130" "130" 20 131 130 true true
run_probe "coverage_short" "v254c_coverage_short" 240 "130,240" "130,240" 20 240 130 true false

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "queue" "all_done" "summary=$SUMMARY end=$END_BJT"
write_status "done" "END_BJT=$END_BJT SUMMARY=$SUMMARY"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "EVENTS=$EVENTS"
  echo "STATUS_JSON=$STATUS_JSON"
} | tee -a "$LOG_DIR/run_info.txt"
EOS

sed -i \
  -e "s#__ROOT__#$ROOT#g" \
  -e "s#__RUN_ID__#$RUN_ID#g" \
  -e "s#__GPU__#$GPU#g" \
  -e "s#__LOG_DIR__#$LOG_DIR#g" \
  -e "s#__PYTHON_BIN__#$PYTHON_BIN#g" \
  -e "s#__BASE_CKPT_DEFAULT__#${BASE_CKPT:-$BASE_CKPT_DEFAULT}#g" \
  -e "s#__BASE_ITER_DEFAULT__#${BASE_ITER:-$BASE_ITER_DEFAULT}#g" \
  -e "s#__V253_CKPT_DEFAULT__#${V253_CKPT:-$V253_CKPT_DEFAULT}#g" \
  "$QUEUE_SCRIPT"
chmod +x "$QUEUE_SCRIPT"

setsid bash "$QUEUE_SCRIPT" > "$NOHUP_LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/launcher.pid"

{
  echo "RUN_ID=$RUN_ID"
  echo "PID=$PID"
  echo "GPU=$GPU"
  echo "LOG_DIR=$LOG_DIR"
  echo "NOHUP_LOG=$NOHUP_LOG"
  echo "STATUS_JSON=$LOG_DIR/status.json"
  echo "QUEUE_SCRIPT=$QUEUE_SCRIPT"
  echo "BASE_CKPT=${BASE_CKPT:-$BASE_CKPT_DEFAULT}"
  echo "BASE_ITER=${BASE_ITER:-$BASE_ITER_DEFAULT}"
  echo "V253_CKPT=${V253_CKPT:-$V253_CKPT_DEFAULT}"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "EXPECTED_APPEND_ONLY_TRAIN_END_BJT=$(TZ=Asia/Shanghai date -d '+55 minutes' '+%F %T BJT')"
  echo "EXPECTED_FULL_QUEUE_END_BJT=$(TZ=Asia/Shanghai date -d '+2 hours 30 minutes' '+%F %T BJT')"
} | tee "$LOG_DIR/launch_info.txt"
