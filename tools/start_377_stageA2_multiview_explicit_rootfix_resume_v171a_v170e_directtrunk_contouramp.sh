#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v171a_v170e_directtrunk_contouramp_screen2500}"
START_CKPT="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v170e_v168a_tagfocus_photo_contour_screen3000/best_ckpt.pth}"
ITERATIONS="${3:-2500}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

exec "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v169a_v168a_hardcontour_boundaryactuator.sh" \
  "$EXP_DIR" \
  "$START_CKPT" \
  "$ITERATIONS" \
  --option stageA_377_multiview_explicit_hq_v170e_v168a_tagfocus_photo_contour_v1 \
  --option stageA_377_multiview_explicit_hq_v171a_v170e_directtrunk_contouramp_v1 \
  "$@"
