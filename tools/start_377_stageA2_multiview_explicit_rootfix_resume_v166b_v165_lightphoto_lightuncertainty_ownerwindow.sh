#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v166b_v165_lightphoto_lightuncertainty_ownerwindow_screen1500}"
START_CKPT="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v165_v164b_framebalanced_camquality_screen6k/best_ckpt.pth}"
ITERATIONS="${3:-1500}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

exec "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v165_v164b_framebalanced_camquality.sh" \
  "$EXP_DIR" \
  "$START_CKPT" \
  "$ITERATIONS" \
  --option stageA_377_multiview_explicit_hq_v166b_v165_lightphoto_lightuncertainty_ownerwindow_v1 \
  "$@"
