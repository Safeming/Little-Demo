#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-stageB_v255_support_audit_queue_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v255_support_audit_queue_${RUN_ID}}"
mkdir -p "$LOG_DIR"

NOHUP_LOG="$LOG_DIR/nohup.log"
QUEUE_SCRIPT="$LOG_DIR/run_v255_support_audit_queue.sh"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"

BASE_V250_EXP="$ROOT/exp/stageB/377_hulk_light_v250b_gapaware_component_support_9h_stageB_v250_gapaware_component_support_9h_20260515_002233_bjt"
BASE_CKPT_DEFAULT="$BASE_V250_EXP/ckpt144950.pth"
BASE_ITER_DEFAULT="144950"

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
    sleep 45
  done
}

run_probe() {
  local name="$1"
  local train_name="$2"
  local render_views="$3"
  local render_frames="$4"
  local split_dir="$5"
  local extra_overrides="$6"
  local local_run_id="${RUN_ID}_${name}"
  local local_log="$LOG_DIR/${name}"
  mkdir -p "$local_log"
  log_event "$name" "start" "render_views=$render_views render_frames=$render_frames split=$split_dir"
  wait_for_gpu "$name"
  env \
    EXPERIMENT_VERSION="v255" \
    EXPERIMENT_SLUG="$name" \
    TRAIN_NAME="$train_name" \
    LOG_PREFIX="377_stageB_v255_${name}" \
    RENDER_TAG="v255" \
    PURPOSE="v255 support audit: short probes to localize component support placement/visibility/generalization break" \
    RUN_ID="$local_run_id" \
    GPU="$GPU" \
    LOG_DIR="$local_log" \
    WAIT_FOR_FREE_GPU=1 \
    DO_RENDER=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    BASE_CKPT="$BASE_CKPT_DEFAULT" \
    BASE_ITER="$BASE_ITER_DEFAULT" \
    STAGEB_ITERATIONS=130 \
    STAGEB_CHECKPOINT_STEPS="130" \
    RENDER_CHECKPOINT_STEPS="130" \
    TEST_INTERVAL=1000 \
    DENSIFY_FROM_STEP=20 \
    DENSIFY_UNTIL_STEP=131 \
    DENSIFY_INTERVAL=130 \
    DENSIFY_GRAD_THRESHOLD=0.00110 \
    TRAIN_VIEWS="[1,2,3,4,5,6,7,8,9,10,11,12]" \
    VAL_VIEWS="[21,22,23]" \
    TEST_VIEWS="[21,22,23]" \
    TRAIN_FRAMES="[0,570,1]" \
    VAL_FRAMES="[0,570,60]" \
    TEST_FRAMES="[0,570,60]" \
    RENDER_TEST_VIEWS="$render_views" \
    RENDER_TEST_FRAMES="$render_frames" \
    RENDER_SPLIT_DIR="$split_dir" \
    BINDING_MAPS="$BINDING_MAPS" \
    MONTAGE_PANELS="$MONTAGE_PANELS" \
    BINDING_DENSIFY_DISABLE_CLONE=true \
    BINDING_DENSIFY_DISABLE_SPLIT=true \
    BOUNDARY_COMPONENT_SUPPORT_COVERAGE_VERBOSE=true \
    BOUNDARY_COMPONENT_SUPPORT_PROJECTION_AUDIT_VERBOSE=true \
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
    EXTRA_TRAIN_OVERRIDES="${extra_overrides}" \
    bash "$ROOT/tools/run_377_stageB_v248_boundary_component_support_serial.sh"
  copy_summary_rows "$local_log/summary.tsv"
  log_event "$name" "done" "summary=$local_log/summary.tsv"
}

write_status "starting" "v255 support audit queue"
{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "GPU=$GPU"
  echo "LOG_DIR=$LOG_DIR"
  echo "BASE_CKPT=$BASE_CKPT_DEFAULT"
  echo "SUMMARY=$SUMMARY"
  echo "EVENTS=$EVENTS"
} | tee "$LOG_DIR/run_info.txt"

# Probe A: current v254-style append-only behavior, render test views to confirm generalization failure with new audit signals.
run_probe \
  "baseline_testview" \
  "v255a_baseline_testview" \
  "[21,22,23]" \
  "[0,570,120]" \
  "test-view" \
  ""

# Probe B: render train/birth-like views. If projection audit is good and train-view overlap is high, the break is cross-view generalization.
run_probe \
  "baseline_trainview" \
  "v255b_baseline_trainview" \
  "[1,2,3,4,5,6]" \
  "[0,570,120]" \
  "test-view" \
  ""

# Probe C: no 3D offset, isolates whether current offset sends children away from useful support.
run_probe \
  "zero_offset_testview" \
  "v255c_zero_offset_testview" \
  "[21,22,23]" \
  "[0,570,120]" \
  "test-view" \
  "++model.gaussian.boundary_component_support_offset_scale=0.0 ++model.gaussian.boundary_component_support_screen_gap_offset_scale=0.0 ++model.gaussian.boundary_component_support_offset_min=0.0"

# Probe D: stronger screen-gap offset, tests whether placement is simply under-shooting the target pixel.
run_probe \
  "strong_offset_testview" \
  "v255d_strong_offset_testview" \
  "[21,22,23]" \
  "[0,570,120]" \
  "test-view" \
  "++model.gaussian.boundary_component_support_offset_scale=0.90 ++model.gaussian.boundary_component_support_screen_gap_offset_scale=1.45 ++model.gaussian.boundary_component_support_screen_gap_ratio_max=3.25 ++model.gaussian.boundary_component_support_offset_max=0.052 ++model.gaussian.boundary_component_support_screen_gap_offset_max=0.052"

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
  -e "s#__BASE_CKPT_DEFAULT__#$BASE_CKPT_DEFAULT#g" \
  -e "s#__BASE_ITER_DEFAULT__#$BASE_ITER_DEFAULT#g" \
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
  echo "QUEUE_SCRIPT=$QUEUE_SCRIPT"
  echo "STATUS_JSON=$LOG_DIR/status.json"
  echo "EVENTS=$LOG_DIR/events.tsv"
  echo "SUMMARY=$LOG_DIR/summary.tsv"
  echo "BASE_CKPT=$BASE_CKPT_DEFAULT"
  echo "BASE_ITER=$BASE_ITER_DEFAULT"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "EXPECTED_AUDIT_END_BJT=$(TZ=Asia/Shanghai date -d '+2 hours' '+%F %T BJT')"
} | tee "$LOG_DIR/launch_info.txt"
