#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-stageC_min_demo_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SOURCE_EXP="${SOURCE_EXP:-$ROOT/exp/stageB/377_hulk_light_v224c_head_reliable_views_preserve_stageB_headfix_fixed_20260511_103654_bjt_v224c_parserhard_rgbclip_best_20260511_131226_bjt}"
OUT_DIR="${OUT_DIR:-$ROOT/exp/stageC/377_stageC_min_demo_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageC/logs/377_stageC_min_demo_${RUN_ID}}"
MASK_SOURCE="${MASK_SOURCE:-compact}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"

SELECT=(
  render_c21_f000240.png
  render_c21_f000300.png
  render_c22_f000240.png
  render_c23_f000300.png
  render_c23_f000420.png
)

mkdir -p "$OUT_DIR" "$LOG_DIR"

for required in "$PYTHON_BIN" "$SOURCE_EXP/test-view/renders" "$SOURCE_EXP/test-view/semantic_editable_assets" "$SOURCE_EXP/test-view/binding_maps" "$COMPACT_MAPPING"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

if [ "$MASK_SOURCE" = "direct_parser" ] && [ ! -e "$PARSER_ROOT/CoreView_377/mask_cihp/Camera_B21/000240.png" ]; then
  echo "missing direct parser test-view masks under $PARSER_ROOT" >&2
  exit 2
fi

{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "SOURCE_EXP=$SOURCE_EXP"
  echo "OUT_DIR=$OUT_DIR"
  echo "MASK_SOURCE=$MASK_SOURCE"
  echo "PARSER_ROOT=$PARSER_ROOT"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "COMPACT_MAPPING=$COMPACT_MAPPING"
} | tee "$LOG_DIR/run_info.txt"

"$PYTHON_BIN" tools/make_377_stageC_editable_demo.py \
  --exp-dir "$SOURCE_EXP" \
  --split test-view \
  --output-dir "$OUT_DIR" \
  --select "${SELECT[@]}" \
  --panel-width 220 \
  --header-height 34 \
  --gap 8 \
  --mask-source "$MASK_SOURCE" \
  --parser-root "$PARSER_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --compact-mapping "$COMPACT_MAPPING" \
  2>&1 | tee "$LOG_DIR/make_demo.log"

{
  echo "END_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "MANIFEST=$OUT_DIR/manifest.json"
  echo "EDIT_STACK=$OUT_DIR/stageC_edit_demo_stacked.png"
  echo "INTERP_STACK=$OUT_DIR/stageC_interpretability_stacked.png"
  echo "REGION_STACK=$OUT_DIR/stageC_region_cutouts_stacked.png"
} | tee -a "$LOG_DIR/run_info.txt"
