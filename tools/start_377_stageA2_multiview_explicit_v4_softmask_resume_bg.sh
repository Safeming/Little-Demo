#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
BASE_EXP="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_v4_rootfix_v2}"
EXP_DIR="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_v4_softmask_resume_v1}"
LOG_DIR="$ROOT/exp/stageA2/logs"
LOG_PATH="${3:-$LOG_DIR/377_multiview_explicit_hq_v4_softmask_resume_v1.log}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"

nohup bash "$ROOT/tools/start_377_stageA2_multiview_explicit_v4_softmask_resume.sh"   "$BASE_EXP"   "$EXP_DIR"   "$@"   >"$LOG_PATH" 2>&1 &

PID=$!
echo "Started StageA2 v4 soft-mask resume"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "Exp: $EXP_DIR"
