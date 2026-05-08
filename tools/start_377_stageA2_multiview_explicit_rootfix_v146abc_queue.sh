#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
QUEUE_ROOT="${1:-$ROOT/exp/comparisons/v146ab_earlyhybrid_screen4k_queue_$(date +%Y%m%d_%H%M%S)}"
ITER_SCREEN="${2:-4000}"
if [ "$#" -ge 2 ]; then
  shift 2
else
  shift "$#"
fi

mkdir -p "$QUEUE_ROOT"
QUEUE_LOG="$QUEUE_ROOT/queue.log"
STATUS_FILE="$QUEUE_ROOT/status.txt"

V146A_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v146a_v136a_earlyfresh_hybridskin_anchor_screen4k"
V146B_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v146b_v136a_earlyfresh_hybridskin_support_screen4k"
SCREEN_SAVE_STEPS="[1000,2000,4000]"
SCREEN_CKPT_STEPS="[4000]"
SCREEN_TEST_STEPS="[1000,2000,4000]"
SCREEN_TEST_INTERVAL="1000"

run_one() {
  local name="$1"
  local exp_dir="$2"
  local iterations="$3"
  shift 3
  local log_file="$exp_dir/train.log"

  mkdir -p "$exp_dir"
  echo "[$(date '+%F %T')] START $name exp=$exp_dir" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
  bash "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_fresh_v146_hybrid.sh" \
    "$exp_dir" \
    "$iterations" \
    "$@" > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE  $name exp=$exp_dir" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
}

echo "queue_root=$QUEUE_ROOT" | tee "$QUEUE_LOG" "$STATUS_FILE"
echo "non_rigid=mlp" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "iter_screen=$ITER_SCREEN" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "save_iterations=$SCREEN_SAVE_STEPS" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "checkpoint_iterations=$SCREEN_CKPT_STEPS" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "test_iterations=$SCREEN_TEST_STEPS" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "test_interval=$SCREEN_TEST_INTERVAL" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "queue_mode=screen_only" | tee -a "$QUEUE_LOG" "$STATUS_FILE"

run_one \
  "v146a_earlyfresh_hybridskin_anchor" \
  "$V146A_EXP" \
  "$ITER_SCREEN" \
  --option stageA_377_multiview_explicit_hq_v146a_v136a_earlyfresh_hybridskin_anchor_v1 \
  --extra-override "save_iterations=$SCREEN_SAVE_STEPS" \
  --extra-override "checkpoint_iterations=$SCREEN_CKPT_STEPS" \
  --extra-override "test_iterations=$SCREEN_TEST_STEPS" \
  --extra-override "test_interval=$SCREEN_TEST_INTERVAL" \
  "$@"

run_one \
  "v146b_earlyfresh_hybridskin_support" \
  "$V146B_EXP" \
  "$ITER_SCREEN" \
  --option stageA_377_multiview_explicit_hq_v146b_v136a_earlyfresh_hybridskin_support_v1 \
  --extra-override "save_iterations=$SCREEN_SAVE_STEPS" \
  --extra-override "checkpoint_iterations=$SCREEN_CKPT_STEPS" \
  --extra-override "test_iterations=$SCREEN_TEST_STEPS" \
  --extra-override "test_interval=$SCREEN_TEST_INTERVAL" \
  "$@"

echo "[$(date '+%F %T')] SCREEN_DONE queue_root=$QUEUE_ROOT" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "[$(date '+%F %T')] awaiting review before 12k continuation or v146c" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "[$(date '+%F %T')] resume_candidate_v146a=$V146A_EXP/ckpt4000.pth" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "[$(date '+%F %T')] resume_candidate_v146b=$V146B_EXP/ckpt4000.pth" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
