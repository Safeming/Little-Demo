#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
STAMP="$(date +%m%d-%H%M)"
EXP_DIR="${1:-$ROOT/exp/stageA/377_multiview_recon_hq_$STAMP}"
shift || true

mkdir -p "$EXP_DIR"
cd "$ROOT"
exec "$PYTHON_BIN" "$ROOT/tools/run_377_stageA_recon_hq.py" \
  --exp-dir "$EXP_DIR" \
  "$@"
