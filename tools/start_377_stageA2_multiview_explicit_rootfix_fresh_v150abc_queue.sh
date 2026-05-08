#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
QUEUE_ROOT="${1:-$ROOT/exp/comparisons/v150abc_forwardtrunk_hashgrid_queue_$(date +%Y%m%d_%H%M%S)}"
ITER_SCREEN="${2:-4000}"
if [ "$#" -ge 2 ]; then
  shift 2
else
  shift "$#"
fi

mkdir -p "$QUEUE_ROOT"
QUEUE_LOG="$QUEUE_ROOT/queue.log"
STATUS_FILE="$QUEUE_ROOT/status.txt"

V150A_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v150a_v136a_forwardtrunk_anchor_support_hashgrid_screen4k"
V150B_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v150b_v150a_boundarytag_regrow_hashgrid_screen4k"
V150C_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v150c_v150b_latelayer_hashgrid_screen4k"
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
  bash "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_fresh_v150_forwardtrunk_hashgrid.sh" \
    "$exp_dir" \
    "$iterations" \
    "$@" > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE  $name exp=$exp_dir" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
}

echo "queue_root=$QUEUE_ROOT" | tee "$QUEUE_LOG" "$STATUS_FILE"
echo "non_rigid=hashgrid" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "iter_screen=$ITER_SCREEN" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "save_iterations=$SCREEN_SAVE_STEPS" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "checkpoint_iterations=$SCREEN_CKPT_STEPS" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "test_iterations=$SCREEN_TEST_STEPS" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "test_interval=$SCREEN_TEST_INTERVAL" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "queue_mode=screen_only_serial" | tee -a "$QUEUE_LOG" "$STATUS_FILE"

run_one \
  "v150a_forwardtrunk_anchor_support_hashgrid" \
  "$V150A_EXP" \
  "$ITER_SCREEN" \
  --option stageA_377_multiview_explicit_hq_v150a_v136a_forwardtrunk_anchor_support_hashgrid_v1 \
  --extra-override "save_iterations=$SCREEN_SAVE_STEPS" \
  --extra-override "checkpoint_iterations=$SCREEN_CKPT_STEPS" \
  --extra-override "test_iterations=$SCREEN_TEST_STEPS" \
  --extra-override "test_interval=$SCREEN_TEST_INTERVAL" \
  "$@"

run_one \
  "v150b_boundarytag_regrow_hashgrid" \
  "$V150B_EXP" \
  "$ITER_SCREEN" \
  --option stageA_377_multiview_explicit_hq_v150b_v150a_boundarytag_regrow_hashgrid_v1 \
  --extra-override "save_iterations=$SCREEN_SAVE_STEPS" \
  --extra-override "checkpoint_iterations=$SCREEN_CKPT_STEPS" \
  --extra-override "test_iterations=$SCREEN_TEST_STEPS" \
  --extra-override "test_interval=$SCREEN_TEST_INTERVAL" \
  "$@"

run_one \
  "v150c_latelayer_hashgrid" \
  "$V150C_EXP" \
  "$ITER_SCREEN" \
  --option stageA_377_multiview_explicit_hq_v150c_v150b_latelayer_hashgrid_v1 \
  --extra-override "save_iterations=$SCREEN_SAVE_STEPS" \
  --extra-override "checkpoint_iterations=$SCREEN_CKPT_STEPS" \
  --extra-override "test_iterations=$SCREEN_TEST_STEPS" \
  --extra-override "test_interval=$SCREEN_TEST_INTERVAL" \
  "$@"

echo "[$(date '+%F %T')] SCREEN_DONE queue_root=$QUEUE_ROOT" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
