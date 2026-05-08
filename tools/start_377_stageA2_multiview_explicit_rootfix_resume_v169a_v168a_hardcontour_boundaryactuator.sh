#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v169a_v168a_hardcontour_boundaryactuator_screen1500}"
START_CKPT="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v168a_v142e_isolated_trunkhf_screen1500/best_ckpt.pth}"
ITERATIONS="${3:-1500}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

exec "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v168a_v142e_isolated_trunkhf.sh" \
  "$EXP_DIR" \
  "$START_CKPT" \
  "$ITERATIONS" \
  --option stageA_377_multiview_explicit_hq_v169a_v168a_hardcontour_boundaryactuator_v1 \
  "$@"
