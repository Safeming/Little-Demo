#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_v4_silhouette_opacity_refine_v1}"
START_CKPT="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_v4_rootfix_v2/best_ckpt.pth}"
LOG_DIR="${3:-$ROOT/exp/stageA2/logs}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

mkdir -p "$EXP_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "$EXP_DIR").log"
setsid bash "$ROOT/tools/start_377_stageA2_multiview_explicit_v4_silhouette_opacity_refine.sh"   "$EXP_DIR"   "$START_CKPT"   "$@" > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "PID: $PID"
echo "LOG: $LOG_FILE"
echo "EXP: $EXP_DIR"
echo "START_CKPT: $START_CKPT"
