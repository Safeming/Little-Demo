#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_v4_softmask_dualmask_resume_v2_detailrecover}"
START_CKPT="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_v4_softmask_dualmask_resume_v1/best_ckpt.pth}"
SOFT_MASK_ROOT="${SOFT_MASK_ROOT:-$ROOT/data/mattes_from_hulk_multiview}"
if [ "$#" -ge 2 ]; then
  shift 2
else
  shift "$#"
fi

mkdir -p "$EXP_DIR"
cd "$ROOT"
exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT/tools/run_377_stageA2_multiview_explicit.py" \
  --exp-dir "$EXP_DIR" \
  --texture shallow_mlp \
  --option stageA_377_multiview_explicit_hq_v4 \
  --option stageA_377_multiview_explicit_hq_v4_fast \
  --option stageA_377_multiview_explicit_hq_v4_softmask_dualmask_resume_v2_detailrecover \
  --extra-override "start_checkpoint=$START_CKPT" \
  --extra-override "+dataset.soft_mask.enable=true" \
  --extra-override "+dataset.soft_mask.root_dir=$SOFT_MASK_ROOT" \
  "$@"
