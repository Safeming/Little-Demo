#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
QUEUE_ROOT="${1:-$ROOT/exp/comparisons/v145ab_hybrid_queue_$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${2:-4000}"
START_CKPT="${3:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v142e_v139a_signed_localrefresh_unsquare_residualpush_resume/best_ckpt.pth}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

mkdir -p "$QUEUE_ROOT"
QUEUE_LOG="$QUEUE_ROOT/queue.log"
STATUS_FILE="$QUEUE_ROOT/status.txt"

run_one() {
  local name="$1"
  local option_name="$2"
  local exp_dir="$3"
  shift 3
  local log_file="$exp_dir/train.log"

  mkdir -p "$exp_dir"
  echo "[$(date '+%F %T')] START $name exp=$exp_dir option=$option_name" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
  bash "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_fresh_v139a_v136a_boundarygeom_errordriven.sh" \
    "$exp_dir" \
    "$START_CKPT" \
    "$ITERATIONS" \
    --option stageA_377_multiview_explicit_hq_v142e_v139a_signed_localrefresh_unsquare_residualpush_v1 \
    --option "$option_name" \
    "$@" > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE  $name exp=$exp_dir" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
}

echo "queue_root=$QUEUE_ROOT" | tee "$QUEUE_LOG" "$STATUS_FILE"
echo "iterations=$ITERATIONS" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
echo "start_ckpt=$START_CKPT" | tee -a "$QUEUE_LOG" "$STATUS_FILE"

run_one \
  "v145a_hybrid_skinning_anchor" \
  "stageA_377_multiview_explicit_hq_v145a_v142e_hybrid_skinning_anchor_v1" \
  "$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v145a_v142e_hybrid_skinning_anchor_resume" \
  "$@"

run_one \
  "v145b_hybrid_skinning_support" \
  "stageA_377_multiview_explicit_hq_v145b_v142e_hybrid_skinning_support_v1" \
  "$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v145b_v142e_hybrid_skinning_support_resume" \
  "$@"

echo "[$(date '+%F %T')] ALL_DONE queue_root=$QUEUE_ROOT" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
