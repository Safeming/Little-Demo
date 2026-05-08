#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
BASE_EXP="${1:-$ROOT/exp/stageA/377_baseline_explicit_mono_v1}"
EXP_DIR="${2:-$ROOT/exp/stageA/377_baseline_explicit_mono_sharp_v1}"
START_CKPT="${START_CKPT:-$BASE_EXP/best_ckpt.pth}"
if [ "$#" -ge 2 ]; then
  shift 2
else
  shift "$#"
fi

mkdir -p "$EXP_DIR"
cd "$ROOT"
exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT/tools/run_377_stageA_baseline_explicit.py"   --exp-dir "$EXP_DIR"   --dataset zjumocap_377_mono_hq   --option stageA_377_baseline_explicit_mono_sharp_v1   --extra-override start_checkpoint="$START_CKPT"   "$@"
