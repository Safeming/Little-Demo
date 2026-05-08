#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v164a_v160a_contouralpha_rootrepair_screen1200}"
START_CKPT="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v160a_v158a_mixedboundary_cfinish_screen2k/best_ckpt.pth}"
ITERATIONS="${3:-1200}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

exec "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v160a_v158a_cfinish.sh" \
  "$EXP_DIR" \
  "$START_CKPT" \
  "$ITERATIONS" \
  --option stageA_377_multiview_explicit_hq_v164a_v160a_contouralpha_rootrepair_v1 \
  "$@"
