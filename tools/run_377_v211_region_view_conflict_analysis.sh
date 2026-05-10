#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SMOKE="${SMOKE:-0}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
V205_TEACHER_LOG="${V205_TEACHER_LOG:-$ROOT/exp/stageA2/logs/v205teacher_20260509_130130_bjt}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageA2/logs/v211_region_view_conflict_$RUN_ID}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$LOG_DIR/v211a_region_view_conflict_map}"

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
cd "$ROOT"

collect_train_render_exps() {
  local files=(
    "$V205_TEACHER_LOG/render_exp_c01_05.txt"
    "$V205_TEACHER_LOG/render_exp_c06_10.txt"
    "$V205_TEACHER_LOG/render_exp_c11_15.txt"
    "$V205_TEACHER_LOG/render_exp_c16_20.txt"
  )
  local out=()
  for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
      echo "missing train-view render list: $file" >&2
      exit 4
    fi
    local exp
    exp="$(cat "$file")"
    if [ ! -d "$exp/test-view/renders" ]; then
      echo "missing train-view render dir: $exp/test-view/renders" >&2
      exit 5
    fi
    out+=("$exp")
  done
  printf '%s\n' "${out[@]}"
}

mapfile -t train_render_exps < <(collect_train_render_exps)
printf '%s\n' "${train_render_exps[@]}" > "$LOG_DIR/v211_train_render_exps.txt"

render_args=("${train_render_exps[@]}")
topk=28
if [ "$SMOKE" = "1" ]; then
  render_args=("${train_render_exps[0]}")
  topk=6
fi

printf 'RUN_ID=%s\nSMOKE=%s\nLOG_DIR=%s\nANALYSIS_DIR=%s\n' "$RUN_ID" "$SMOKE" "$LOG_DIR" "$ANALYSIS_DIR" \
  | tee "$LOG_DIR/run_info.txt"

"$PYTHON_BIN" tools/analyze_377_region_view_conflict_map.py \
  --render-exp "${render_args[@]}" \
  --dataset-root "$DATA_ROOT" \
  --parser-root "$PARSER_ROOT" \
  --subject CoreView_377 \
  --split-dir test-view \
  --out-dir "$ANALYSIS_DIR" \
  --band-width 7 \
  --region-erode 5 \
  --min-region-pixels 96 \
  --reliable-l1-thresh 0.075 \
  --missing-hf-ratio 0.78 \
  --topk "$topk" > "$LOG_DIR/v211a_region_view_conflict_map.log" 2>&1

echo "SUMMARY=$ANALYSIS_DIR/summary.md"
echo "TRAINING_PLAN=$ANALYSIS_DIR/region_training_plan.json"
echo "MONTAGE=$ANALYSIS_DIR/region_conflict_montage.png"
