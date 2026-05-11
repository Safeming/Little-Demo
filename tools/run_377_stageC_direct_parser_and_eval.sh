#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-stageC_direct_parser_eval_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SOURCE_EXP="${SOURCE_EXP:-$ROOT/exp/stageB/377_hulk_light_v224c_head_reliable_views_preserve_stageB_headfix_fixed_20260511_103654_bjt_v224c_parserhard_rgbclip_best_20260511_131226_bjt}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
OUT_DIR="${OUT_DIR:-$ROOT/exp/stageC/377_stageC_direct_parser_${RUN_ID}}"
EVAL_DIR="${EVAL_DIR:-$ROOT/exp/stageC/eval/377_compact_vs_hulk_parser_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageC/logs/377_stageC_direct_parser_eval_${RUN_ID}}"

SELECT=(
  render_c21_f000240.png
  render_c21_f000300.png
  render_c22_f000240.png
  render_c23_f000300.png
  render_c23_f000420.png
)

mkdir -p "$OUT_DIR" "$EVAL_DIR" "$LOG_DIR"

for required in \
  "$PYTHON_BIN" \
  "$SOURCE_EXP/test-view/renders" \
  "$SOURCE_EXP/test-view/semantic_editable_assets/compact_head_masks" \
  "$PARSER_ROOT/CoreView_377/mask_cihp/Camera_B21/000240.png" \
  "$PARSER_ROOT/CoreView_377/mask_cihp/Camera_B22/000240.png" \
  "$PARSER_ROOT/CoreView_377/mask_cihp/Camera_B23/000300.png" \
  "$COMPACT_MAPPING"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "SOURCE_EXP=$SOURCE_EXP"
  echo "PARSER_ROOT=$PARSER_ROOT"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "COMPACT_MAPPING=$COMPACT_MAPPING"
  echo "OUT_DIR=$OUT_DIR"
  echo "EVAL_DIR=$EVAL_DIR"
} | tee "$LOG_DIR/run_info.txt"

"$PYTHON_BIN" tools/make_377_stageC_editable_demo.py \
  --exp-dir "$SOURCE_EXP" \
  --split test-view \
  --output-dir "$OUT_DIR" \
  --select "${SELECT[@]}" \
  --mask-source direct_parser \
  --parser-root "$PARSER_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --compact-mapping "$COMPACT_MAPPING" \
  --panel-width 220 \
  --header-height 34 \
  --gap 8 \
  2>&1 | tee "$LOG_DIR/direct_parser_stageC.log"

"$PYTHON_BIN" tools/eval_377_stageB_compact_vs_hulk_parser.py \
  --exp-dir "$SOURCE_EXP" \
  --split test-view \
  --parser-root "$PARSER_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --compact-mapping "$COMPACT_MAPPING" \
  --out-dir "$EVAL_DIR" \
  --select "${SELECT[@]}" \
  --panel-width 190 \
  --header-height 30 \
  --gap 6 \
  2>&1 | tee "$LOG_DIR/compact_vs_parser_eval.log"

{
  echo "END_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "DIRECT_STAGEC=$OUT_DIR"
  echo "DIRECT_EDIT_STACK=$OUT_DIR/stageC_edit_demo_stacked.png"
  echo "DIRECT_REGION_STACK=$OUT_DIR/stageC_region_cutouts_stacked.png"
  echo "EVAL_SUMMARY=$EVAL_DIR/summary.json"
  echo "EVAL_METRICS=$EVAL_DIR/per_frame_region_metrics.tsv"
  echo "EVAL_PANEL=$EVAL_DIR/compact_vs_parser_stacked.png"
} | tee -a "$LOG_DIR/run_info.txt"
