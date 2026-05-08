#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v164b_v164a_texture_restore_contourlock_screen2k}"
START_CKPT="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v164a_v160a_contouralpha_rootrepair_screen1200/best_ckpt.pth}"
ITERATIONS="${3:-2000}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

exec "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v164a_v160a_contouralpha.sh" \
  "$EXP_DIR" \
  "$START_CKPT" \
  "$ITERATIONS" \
  --option stageA_377_multiview_explicit_hq_v164b_v164a_texture_restore_contourlock_v1 \
  "$@"
