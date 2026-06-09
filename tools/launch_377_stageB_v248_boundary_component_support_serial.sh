#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-stageB_v248_boundary_component_support_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v248_boundary_component_support_${RUN_ID}}"
mkdir -p "$LOG_DIR"

NOHUP_LOG="$LOG_DIR/nohup.log"

env \
  RUN_ID="$RUN_ID" \
  GPU="$GPU" \
  WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}" \
  DO_RENDER="${DO_RENDER:-1}" \
  PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}" \
  V248_ITERATIONS="${V248_ITERATIONS:-360}" \
  V248_CHECKPOINT_STEPS="${V248_CHECKPOINT_STEPS:-120,240,360}" \
  setsid bash "$ROOT/tools/run_377_stageB_v248_boundary_component_support_serial.sh" \
  > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "$PID" > "$LOG_DIR/launcher.pid"

{
  echo "RUN_ID=$RUN_ID"
  echo "PID=$PID"
  echo "GPU=$GPU"
  echo "LOG_DIR=$LOG_DIR"
  echo "NOHUP_LOG=$NOHUP_LOG"
  echo "STATUS_JSON=$LOG_DIR/status.json"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
} | tee "$LOG_DIR/launch_info.txt"
