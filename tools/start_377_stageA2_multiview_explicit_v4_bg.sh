#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_v4}"
LOG_DIR="${2:-$ROOT/exp/stageA2/logs}"
if [ "$#" -ge 2 ]; then
  shift 2
else
  shift "$#"
fi

mkdir -p "$EXP_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "$EXP_DIR").log"
nohup bash "$ROOT/tools/start_377_stageA2_multiview_explicit_v4.sh" \
  "$EXP_DIR" \
  "$@" > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo "LOG: $LOG_FILE"
echo "EXP: $EXP_DIR"
