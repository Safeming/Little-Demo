#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
RUN_ID="${RUN_ID:-v374_v376_queue_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v374_v376_queue_${RUN_ID}}"
EVENTS="$QUEUE_LOG_DIR/events.tsv"
SUMMARY="$QUEUE_LOG_DIR/summary.tsv"

mkdir -p "$QUEUE_LOG_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'experiment\tstatus\tlog_dir\ttrain_pid\ttrain_exp_dir\ttrain_end_bjt\n' > "$SUMMARY"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

extract_value() {
  local key="$1"
  local file="$2"
  awk -F= -v key="$key" '$1 == key {print substr($0, length(key) + 2)}' "$file" | tail -n 1
}

wait_for_train_if_any() {
  local label="$1"
  local output_log="$2"
  local train_pid
  train_pid="$(extract_value TRAIN_PID "$output_log")"
  if [ -n "$train_pid" ] && kill -0 "$train_pid" 2>/dev/null; then
    log_event "${label}_train_wait_start" "pid=$train_pid"
    while kill -0 "$train_pid" 2>/dev/null; do
      sleep 60
    done
    log_event "${label}_train_wait_done" "pid=$train_pid"
  else
    log_event "${label}_train_wait_skip" "pid=$train_pid"
  fi
}

run_experiment() {
  local label="$1"
  local script="$2"
  local output_log="$QUEUE_LOG_DIR/${label}.launcher.log"
  shift 2
  log_event "${label}_start" "$script"
  env \
    GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
    RUN_ID="${label}_${RUN_ID}" \
    "$@" \
    "$script" \
    > "$output_log" 2>&1
  local log_dir status train_pid train_exp_dir train_end
  log_dir="$(extract_value LOG_DIR "$output_log")"
  status="$(extract_value RAW_GATE_STATUS "$output_log")"
  train_pid="$(extract_value TRAIN_PID "$output_log")"
  train_exp_dir="$(extract_value TRAIN_EXP_DIR "$output_log")"
  train_end="$(extract_value TRAIN_EST_END_BJT "$output_log")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "${status:-done}" "$log_dir" "$train_pid" "$train_exp_dir" "$train_end" >> "$SUMMARY"
  log_event "${label}_launcher_done" "status=${status:-done} log=$output_log"
  wait_for_train_if_any "$label" "$output_log"
}

START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d '+3 hours 10 minutes' '+%F %T BJT')"
cat > "$QUEUE_LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
SUMMARY=$SUMMARY
QUEUE_LOG_DIR=$QUEUE_LOG_DIR

Sequence:
  v374 portfolio merge raw gate, train only on strict_pass.
  v375 c23 no-gain diagnostic, no train.
  v376 v371 portfolio plus signed outer shrink raw gate, train only on strict_pass.
EOF

log_event "queue_start" "est_end=$EST_END_BJT"

run_experiment v374 "$ROOT/tools/run_377_explicit_binding_v374_portfolio_merge_grouped_actuator.sh"
run_experiment v375 "$ROOT/tools/run_377_explicit_binding_v375_c23_no_gain_diagnostic.sh"
run_experiment v376 "$ROOT/tools/run_377_explicit_binding_v376_outer_shrink_portfolio_grouped_actuator.sh"

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "queue_done" "$END_BJT"
echo "QUEUE_LOG_DIR=$QUEUE_LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "START_BJT=$START_BJT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
