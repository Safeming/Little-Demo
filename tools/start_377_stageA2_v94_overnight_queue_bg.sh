#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/exp/stageA2/logs"
QUEUE_LOG="${1:-$LOG_DIR/v94_overnight_queue_${STAMP}.log}"
OUTPUT_ROOT="${2:-$ROOT/exp/comparisons/v94_overnight_${STAMP}}"
WORKERS="${WORKERS:-2}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "[wrapper] started_at=$(TZ=Asia/Shanghai date '+%F %T %Z')"
echo "[wrapper] queue_log=$QUEUE_LOG"
echo "[wrapper] output_root=$OUTPUT_ROOT"
echo "[wrapper] workers=$WORKERS"

exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT/tools/run_377_stageA2_v94_overnight_queue.py" \
  --output-root "$OUTPUT_ROOT" \
  --log-dir "$LOG_DIR" \
  --workers "$WORKERS"
