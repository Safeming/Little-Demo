#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-stageC_v233d_softer_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_shoes_preserve_control_stageB_compact_v233_skincloth_20260512_161652_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt135710.pth}"

RENDER_EXP="${RENDER_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_stageC_softer_${RUN_ID}_render_compact_final}"
OUT_DIR="${OUT_DIR:-$ROOT/exp/stageC/377_stageC_v233d_softer_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageC/logs/377_stageC_v233d_softer_${RUN_ID}}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

SELECT=(
  render_c21_f000240.png
  render_c21_f000300.png
  render_c22_f000240.png
  render_c23_f000300.png
  render_c23_f000420.png
)
BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin]"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "GPU=$GPU"
  echo "BASE_EXP=$BASE_EXP"
  echo "BASE_CKPT=$BASE_CKPT"
  echo "RENDER_EXP=$RENDER_EXP"
  echo "OUT_DIR=$OUT_DIR"
  echo "PARSER_ROOT=$PARSER_ROOT"
  echo "COMPACT_MAPPING=$COMPACT_MAPPING"
  echo "SOFTER_EXPORT=opacity_threshold=0.025 close_kernel=5 erode_kernel=1"
} | tee "$LOG_DIR/run_info.txt"

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64"
)

env "${COMMON_ENV[@]}" "$PYTHON_BIN" "$ROOT/render.py" \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  mode=test \
  "load_ckpt=$BASE_CKPT" \
  "exp_dir=$RENDER_EXP" \
  "dataset.root_dir=$DATA_ROOT" \
  "dataset.preload=false" \
  "dataset.test_views.view=[21,22,23]" \
  "dataset.test_frames.view=[0,570,60]" \
  "dataset.parsing_prior.enable=true" \
  "dataset.parsing_prior.roi_enable=true" \
  "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
  "dataset.parsing_prior.parser_layout=cihp_subject" \
  "dataset.parsing_prior.use_direct_parser_labels=true" \
  "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING" \
  "export_interpretability=true" \
  "++binding_map_names=$BINDING_MAPS" \
  "++binding_map_use_opacity_mask=true" \
  "++binding_map_hard_fg_opacity_threshold=0.030" \
  "++binding_map_opacity_threshold=0.030" \
  "++binding_map_compact_semantic_opacity_threshold=0.025" \
  "++binding_map_hard_fg_close_kernel=5" \
  "++binding_map_hard_fg_erode_kernel=1" \
  "++binding_map_support_close_kernel=5" \
  "++binding_map_support_erode_kernel=1" \
  "++render_export_opacity_threshold=0.025" \
  "++render_export_close_kernel=5" \
  "++render_export_erode_kernel=1" \
  "export_semantic_editable_assets=true" \
  "semantic_editable_parser_root=$PARSER_ROOT" \
  "semantic_editable_parser_layout=cihp_subject" \
  "semantic_editable_direct_parser_mode=false" \
  "semantic_editable_export_compact_head=true" \
  "semantic_editable_include_binding_summary=true" \
  "++semantic_editable_compact_opacity_threshold=0.025" \
  "++semantic_editable_compact_confidence_threshold=0.0" \
  "+semantic_editable_preview_min_area=18" \
  "hydra.run.dir=$LOG_DIR/hydra_render" \
  "wandb_disable=true" \
  2>&1 | tee "$LOG_DIR/render_softer.log"

"$PYTHON_BIN" "$ROOT/tools/eval_377_stageB_compact_vs_hulk_parser.py" \
  --exp-dir "$RENDER_EXP" \
  --split test-view \
  --parser-root "$PARSER_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --compact-mapping "$COMPACT_MAPPING" \
  --out-dir "$RENDER_EXP/diagnostics/compact_vs_hulk_parser" \
  --select "${SELECT[@]}" \
  --panel-width 190 \
  --header-height 30 \
  --gap 6 \
  2>&1 | tee "$LOG_DIR/eval_compact_vs_parser.log" || true

"$PYTHON_BIN" "$ROOT/tools/make_377_stageC_editable_demo.py" \
  --exp-dir "$RENDER_EXP" \
  --split test-view \
  --output-dir "$OUT_DIR" \
  --select "${SELECT[@]}" \
  --panel-width 220 \
  --header-height 34 \
  --gap 8 \
  --mask-source compact \
  --parser-root "$PARSER_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --compact-mapping "$COMPACT_MAPPING" \
  2>&1 | tee "$LOG_DIR/make_stageC_demo.log" || true

"$PYTHON_BIN" "$ROOT/tools/make_binding_paper_montage.py" \
  --exp-dir "$RENDER_EXP" \
  --gt-root "$DATA_ROOT/CoreView_377" \
  --split test-view \
  --panels gt render layer region body_prob cloth_prob compact_semantic thin semantic \
  --select "${SELECT[@]}" \
  --output-dir "$RENDER_EXP/test-view/paper_montages_selected" \
  2>&1 | tee "$LOG_DIR/make_binding_montage.log" || true

"$PYTHON_BIN" "$ROOT/tools/analyze_377_render_contours.py" \
  --render-exp "$RENDER_EXP" \
  --dataset-root "$DATA_ROOT" \
  --subject CoreView_377 \
  --band-width 7 \
  --topk 12 \
  --out-dir "$RENDER_EXP/diagnostics/contours" \
  2>&1 | tee "$LOG_DIR/analyze_contours.log" || true

{
  echo "END_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "RENDER_EXP=$RENDER_EXP"
  echo "OUT_DIR=$OUT_DIR"
  echo "EDIT_STACK=$OUT_DIR/stageC_edit_demo_stacked.png"
  echo "REGION_STACK=$OUT_DIR/stageC_region_cutouts_stacked.png"
  echo "INTERP_STACK=$OUT_DIR/stageC_interpretability_stacked.png"
} | tee -a "$LOG_DIR/run_info.txt"
