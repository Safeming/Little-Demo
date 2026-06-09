#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
RUN_ID="${RUN_ID:-v381_closure_raw_selector_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v381_closure_raw_selector_${RUN_ID}}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v381_closure_raw_selector_${RUN_ID}}"
LAUNCHER_LOG="$LOG_DIR/v381_closure_raw_selector.launcher.log"

mkdir -p "$LOG_DIR" "$EXP_ROOT"

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "EXP_ROOT=$EXP_ROOT"
echo "LAUNCHER_LOG=$LAUNCHER_LOG"

env \
  GPU="$GPU" \
  PYTHON_BIN="$PYTHON_BIN" \
  CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  RUN_ID="$RUN_ID" \
  LOG_DIR="$LOG_DIR" \
  EXP_ROOT="$EXP_ROOT" \
  "$PYTHON_BIN" "$ROOT/tools/run_377_explicit_binding_v381_closure_raw_selector.py" \
  --run-id "$RUN_ID" \
  --log-dir "$LOG_DIR" \
  --exp-root "$EXP_ROOT" \
  > "$LAUNCHER_LOG" 2>&1
