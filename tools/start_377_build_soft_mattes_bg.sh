#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
SUBJECT="${1:-CoreView_377}"
CAMERAS="${2:-1-20}"
LOG_DIR="$ROOT/exp/stageA2/logs"
LOG_PATH="${3:-$LOG_DIR/${SUBJECT}_soft_matte_build.log}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT/tools/build_soft_mattes_from_hulk_parser.py"   --subject "$SUBJECT"   --cameras "$CAMERAS"   --overwrite   >"$LOG_PATH" 2>&1 &

PID=$!
echo "Started soft matte build for $SUBJECT (cameras: $CAMERAS)"
echo "PID: $PID"
echo "Log: $LOG_PATH"
