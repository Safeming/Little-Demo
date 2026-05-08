#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_v4_texture_refine_v1}"
START_CKPT="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_v4_rootfix_v2/best_ckpt.pth}"
if [ "$#" -ge 2 ]; then
  shift 2
else
  shift "$#"
fi

cd "$ROOT"
"$PYTHON_BIN" "$ROOT/tools/run_377_stageA2_multiview_explicit.py"   --exp-dir "$EXP_DIR"   --texture shallow_mlp   --option stageA_377_multiview_explicit_hq_v4   --option stageA_377_multiview_explicit_hq_v4_texture_refine_v1   --extra-override "start_checkpoint=$START_CKPT"   "$@"
