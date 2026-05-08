#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
V155A_EXP="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v155a_v152a_stagedlayer_geommain_screen4k}"
V155B_EXP="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v155b_v155a_ownerhf_handoff_cleanup_screen3k}"
V155A_ITERS="${3:-4000}"
V155B_ITERS="${4:-3000}"

mkdir -p "$(dirname "$V155A_EXP")" "$(dirname "$V155B_EXP")"

echo "[v155ab] stage1 start: $(date '+%F %T %Z')"
bash "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_fresh_v155a_stagedlayer_geommain.sh" \
  "$V155A_EXP" \
  "$V155A_ITERS"

V155A_CKPT="$V155A_EXP/best_ckpt.pth"
if [ ! -f "$V155A_CKPT" ]; then
  echo "[v155ab] missing stage1 checkpoint: $V155A_CKPT" >&2
  exit 1
fi

echo "[v155ab] stage2 start: $(date '+%F %T %Z')"
bash "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v155b_ownerhf_handoff_cleanup.sh" \
  "$V155B_EXP" \
  "$V155A_CKPT" \
  "$V155B_ITERS"

echo "[v155ab] done: $(date '+%F %T %Z')"
