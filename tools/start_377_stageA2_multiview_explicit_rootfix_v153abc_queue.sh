#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
QUEUE_ROOT="${1:-$ROOT/exp/comparisons/v153abc_resume_queue_$(date +%Y%m%d_%H%M%S)}"
V152A_CKPT="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v152a_v151_geomlearn_probe_screen4k/best_ckpt.pth}"
if [ "$#" -ge 2 ]; then
  shift 2
else
  shift "$#"
fi

mkdir -p "$QUEUE_ROOT"
QUEUE_LOG="$QUEUE_ROOT/queue.log"
STATUS_FILE="$QUEUE_ROOT/status.txt"

V153A_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v153a_v152a_ownerhf_handoff_screen4k"
V153B_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v153b_v152a_cleanup_signedprune_screen2p5k"
V153C_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v153c_v152a_ownerhf_cleanup_screen3k"

run_one() {
  local name="$1"
  local exp_dir="$2"
  local start_ckpt="$3"
  local iterations="$4"
  shift 4
  local log_file="$exp_dir/train.log"

  mkdir -p "$exp_dir"
  echo "[$(date '+%F %T')] START $name exp=$exp_dir ckpt=$start_ckpt" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
  bash "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v153_stack.sh" \
    "$exp_dir" \
    "$start_ckpt" \
    "$iterations" \
    "$@" > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE  $name exp=$exp_dir" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
}

echo "queue_root=$QUEUE_ROOT" | tee "$QUEUE_LOG" "$STATUS_FILE"
echo "queue_mode=serial_single_gpu" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "resume_source_v152a=$V152A_CKPT" | tee -a "$QUEUE_LOG" "$STATUS_FILE"

run_one \
  "v153a_ownerhf_handoff" \
  "$V153A_EXP" \
  "$V152A_CKPT" \
  "4000" \
  --option stageA_377_multiview_explicit_hq_v153a_v152a_ownerhf_handoff_v1 \
  "$@"

run_one \
  "v153b_cleanup_signedprune" \
  "$V153B_EXP" \
  "$V152A_CKPT" \
  "2500" \
  --option stageA_377_multiview_explicit_hq_v153b_v152a_cleanup_signedprune_v1 \
  "$@"

run_one \
  "v153c_ownerhf_cleanup" \
  "$V153C_EXP" \
  "$V153A_EXP/best_ckpt.pth" \
  "3000" \
  --option stageA_377_multiview_explicit_hq_v153c_v152a_ownerhf_cleanup_v1 \
  "$@"

echo "[$(date '+%F %T')] ALL_DONE queue_root=$QUEUE_ROOT" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
