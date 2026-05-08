#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_v4}"
shift || true

cd "$ROOT"
"$PYTHON_BIN" "$ROOT/tools/run_377_stageA2_multiview_explicit.py" \
  --exp-dir "$EXP_DIR" \
  --texture mlp \
  --option stageA_377_multiview_explicit_hq_v4 \
  "$@"
