#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
EXP_DIR="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v181a_v179c_camera_affine_screen2500"
START_CKPT="$ROOT/exp/377_multiview_explicit_hq_rootfix_resume_v179c_v178c_k6_camquality_tightphoto_screen3000/best_ckpt.pth"
ITERATIONS="2500"

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  EXP_DIR="$1"
  shift
fi
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  START_CKPT="$1"
  shift
fi
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  ITERATIONS="$1"
  shift
fi

mkdir -p "$EXP_DIR"
cd "$ROOT"

exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT/tools/run_377_stageA2_multiview_explicit.py" \
  --exp-dir "$EXP_DIR" \
  --python "$PYTHON_BIN" \
  --texture shallow_mlp_lownoise \
  --option stageA_377_multiview_explicit_hq_v170e_v168a_tagfocus_photo_contour_v1 \
  --option stageA_377_multiview_explicit_hq_v175a_v170e_hfsource_edgesupervise_v1 \
  --option stageA_377_multiview_explicit_hq_v177a_v176a_sameframe_k4_v1 \
  --option stageA_377_multiview_explicit_hq_v179c_v178c_k6_camquality_tightphoto_v1 \
  --option stageA_377_multiview_explicit_hq_v181a_v179c_camera_affine_v1 \
  --extra-override "start_checkpoint=$START_CKPT" \
  --extra-override "opt.iterations=$ITERATIONS" \
  --extra-override "++validation_image_log_limit=0" \
  "$@"
