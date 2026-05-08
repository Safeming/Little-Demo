#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
LOG_DIR="$ROOT/exp/stageA2/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"

CURRENT_EXP="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v157c1_v157b3_signedfinishing_screen2k}"
C2_EXP="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v157c2_v157b3_signedsoft_screen1p5k}"
C3_EXP="${3:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v157c3_v157b3_boundaryhead_screen1p5k}"
C4_EXP="${4:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v157c4_v157b3_facehfpolish_screen1p5k}"
C2_ITERS="${5:-1500}"
C3_ITERS="${6:-1500}"
C4_ITERS="${7:-1500}"
if [ "$#" -ge 7 ]; then
  shift 7
else
  shift "$#"
fi

mkdir -p "$LOG_DIR" "$(dirname "$C2_EXP")"
QUEUE_LOG="$LOG_DIR/v157c_queue_${STAMP}.log"
STAGE_C2_LOG="$LOG_DIR/v157c2_${STAMP}.log"
STAGE_C3_LOG="$LOG_DIR/v157c3_${STAMP}.log"
STAGE_C4_LOG="$LOG_DIR/v157c4_${STAMP}.log"

log_msg() {
  echo "[$(date '+%F %T %Z')] $1" | tee -a "$QUEUE_LOG"
}

wait_current_finish() {
  local target="$1"
  while pgrep -f "$target" >/dev/null 2>&1; do
    log_msg "waiting current_run=$target"
    sleep 60
  done
}

run_stage() {
  local name="$1"
  local script="$2"
  local log_file="$3"
  shift 3
  log_msg "$name start exp=$1"
  bash "$script" "$@" > "$log_file" 2>&1
  log_msg "$name done log=$log_file"
}

log_msg "queue start"
log_msg "current_exp=$CURRENT_EXP"
wait_current_finish "$CURRENT_EXP"

run_stage \
  "v157c2" \
  "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v157c2_v157b3_signedsoft.sh" \
  "$STAGE_C2_LOG" \
  "$C2_EXP" \
  "$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v157b3_v157a_ownerhf_midhandoff_screen3k/best_ckpt.pth" \
  "$C2_ITERS" \
  "$@"

run_stage \
  "v157c3" \
  "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v157c3_v157b3_boundaryhead.sh" \
  "$STAGE_C3_LOG" \
  "$C3_EXP" \
  "$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v157b3_v157a_ownerhf_midhandoff_screen3k/best_ckpt.pth" \
  "$C3_ITERS" \
  "$@"

run_stage \
  "v157c4" \
  "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v157c4_v157b3_facehfpolish.sh" \
  "$STAGE_C4_LOG" \
  "$C4_EXP" \
  "$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v157b3_v157a_ownerhf_midhandoff_screen3k/best_ckpt.pth" \
  "$C4_ITERS" \
  "$@"

log_msg "queue done"
log_msg "stage_logs=$STAGE_C2_LOG,$STAGE_C3_LOG,$STAGE_C4_LOG"
