#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v167a_v142e_trunkhf_localcolor_reanchor_screen2k}"
START_CKPT="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v142e_v139a_signed_localrefresh_unsquare_residualpush_resume/best_ckpt.pth}"
ITERATIONS="${3:-2000}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

exec "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_fresh_v139a_v136a_boundarygeom_errordriven.sh" \
  "$EXP_DIR" \
  "$START_CKPT" \
  "$ITERATIONS" \
  --option stageA_377_multiview_explicit_hq_v142e_v139a_signed_localrefresh_unsquare_residualpush_v1 \
  --option stageA_377_multiview_explicit_hq_v167a_v142e_trunkhf_localcolor_reanchor_v1 \
  "$@"
