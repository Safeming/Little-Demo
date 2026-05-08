#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
LOG_DIR="${1:-$ROOT/exp/stageA2/logs}"
if [ "$#" -ge 1 ]; then
  shift 1
fi

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/v157c_queue_bg_${STAMP}.log"

setsid bash "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_v157c_overnight_queue.sh" \
  "$@" > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "PID: $PID"
echo "LOG: $LOG_FILE"
