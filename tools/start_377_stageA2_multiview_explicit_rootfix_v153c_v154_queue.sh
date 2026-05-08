#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
QUEUE_ROOT="$ROOT/exp/comparisons/v153c_v154_queue_$(date +%Y%m%d_%H%M%S)"
V153A_CKPT="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v153a_v152a_ownerhf_handoff_screen4k/best_ckpt.pth"
V153C_CKPT="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v153c_v152a_ownerhf_cleanup_screen3k/best_ckpt.pth"

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  QUEUE_ROOT="$1"
  shift
fi

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  V153A_CKPT="$1"
  shift
fi

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  V153C_CKPT="$1"
  shift
fi

mkdir -p "$QUEUE_ROOT"
QUEUE_LOG="$QUEUE_ROOT/queue.log"
STATUS_FILE="$QUEUE_ROOT/status.txt"

V153C_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v153c_v152a_ownerhf_cleanup_screen3k"
V154A_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v154a_v153a_ownerregrow_support_screen3k"
V154B_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v154b_v153a_ownerhf_boost_screen3k"
V154C_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v154c_v153a_ownerregrow_hfboost_screen3k"

run_one() {
  local name="$1"
  local exp_dir="$2"
  local start_ckpt="$3"
  local iterations="$4"
  shift 4
  local log_file="$exp_dir/train.log"

  mkdir -p "$exp_dir"
  echo "[$(date '+%F %T')] START $name exp=$exp_dir ckpt=$start_ckpt" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
  echo "[$(date '+%F %T')] RESUME $name" >> "$log_file"
  bash "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v153_stack.sh" \
    "$exp_dir" \
    "$start_ckpt" \
    "$iterations" \
    "$@" >> "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE  $name exp=$exp_dir" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
}

echo "queue_root=$QUEUE_ROOT" | tee "$QUEUE_LOG" "$STATUS_FILE"
echo "queue_mode=serial_single_gpu" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "resume_source_v153a=$V153A_CKPT" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "resume_source_v153c=$V153C_CKPT" | tee -a "$QUEUE_LOG" "$STATUS_FILE"

run_one \
  "v153c_ownerhf_cleanup_resume" \
  "$V153C_EXP" \
  "$V153C_CKPT" \
  "3000" \
  --option stageA_377_multiview_explicit_hq_v153c_v152a_ownerhf_cleanup_v1 \
  "$@"

run_one \
  "v154a_ownerregrow_support" \
  "$V154A_EXP" \
  "$V153A_CKPT" \
  "3000" \
  --option stageA_377_multiview_explicit_hq_v153a_v152a_ownerhf_handoff_v1 \
  --option stageA_377_multiview_explicit_hq_v154a_v153a_ownerregrow_support_v1 \
  "$@"

run_one \
  "v154b_ownerhf_boost" \
  "$V154B_EXP" \
  "$V153A_CKPT" \
  "3000" \
  --option stageA_377_multiview_explicit_hq_v153a_v152a_ownerhf_handoff_v1 \
  --option stageA_377_multiview_explicit_hq_v154b_v153a_ownerhf_boost_v1 \
  "$@"

run_one \
  "v154c_ownerregrow_hfboost" \
  "$V154C_EXP" \
  "$V153A_CKPT" \
  "3000" \
  --option stageA_377_multiview_explicit_hq_v153a_v152a_ownerhf_handoff_v1 \
  --option stageA_377_multiview_explicit_hq_v154c_v153a_ownerregrow_hfboost_v1 \
  "$@"

echo "[$(date '+%F %T')] ALL_DONE queue_root=$QUEUE_ROOT" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
