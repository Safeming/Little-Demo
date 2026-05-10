#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RENDER_EXP="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v193a_v192a_adaptive_stable_texture_20260508_v193a_adaptive_stable_texture_render_quick}"
OUT_DIR="${2:-$ROOT/exp/stageA2/diagnostics/v194_alignment_shift_gate_v193a}"

cd "$ROOT"
mkdir -p "$OUT_DIR"

exec "$PYTHON_BIN" tools/analyze_377_alignment_shift_upper_bound.py \
  --render-exp "$RENDER_EXP" \
  --dataset-root "$ROOT/data/ZJUMoCap" \
  --subject CoreView_377 \
  --split-dir test-view \
  --band-width 2 \
  --max-shift 4 \
  --step 1 \
  --analysis-scale 0.25 \
  --edge-weight 0.015 \
  --interior-penalty 0.60 \
  --topk 16 \
  --out-dir "$OUT_DIR"
