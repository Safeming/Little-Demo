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
RUN_ID="${RUN_ID:-rgbclip_parserhard_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v224c_head_reliable_views_preserve_stageB_headfix_fixed_20260511_103654_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/best_ckpt.pth}"
BASE_RENDER="${BASE_RENDER:-${BASE_EXP}_render_parserhard_best}"
OUT_EXP="${OUT_EXP:-${BASE_EXP}_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/render_377_stageB_${RUN_ID}}"
MASK_SOURCE="${MASK_SOURCE:-parser_hard}"
MASK_ERODE_KERNEL="${MASK_ERODE_KERNEL:-3}"
HEAD_SELECT=(render_c21_f000240.png render_c21_f000300.png render_c22_f000240.png render_c23_f000300.png render_c23_f000420.png)

mkdir -p "$LOG_DIR"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

echo "RUN_ID=$RUN_ID" | tee "$LOG_DIR/run_info.txt"
echo "BASE_EXP=$BASE_EXP" | tee -a "$LOG_DIR/run_info.txt"
echo "BASE_CKPT=$BASE_CKPT" | tee -a "$LOG_DIR/run_info.txt"
echo "OUT_EXP=$OUT_EXP" | tee -a "$LOG_DIR/run_info.txt"
echo "MASK_SOURCE=$MASK_SOURCE" | tee -a "$LOG_DIR/run_info.txt"
echo "MASK_ERODE_KERNEL=$MASK_ERODE_KERNEL" | tee -a "$LOG_DIR/run_info.txt"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" render.py \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  mode=test \
  "load_ckpt=$BASE_CKPT" \
  "exp_dir=$OUT_EXP" \
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
  "++binding_map_names=[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin]" \
  "++binding_map_use_opacity_mask=true" \
  "++binding_map_hard_fg_opacity_threshold=0.105" \
  "++binding_map_opacity_threshold=0.105" \
  "++binding_map_compact_semantic_opacity_threshold=0.105" \
  "++binding_map_hard_fg_close_kernel=1" \
  "++binding_map_hard_fg_erode_kernel=3" \
  "++binding_map_support_close_kernel=1" \
  "++binding_map_support_erode_kernel=3" \
  "++binding_map_mask_source=parser_hard" \
  "++binding_map_mask_erode_kernel=3" \
  "++render_export_opacity_threshold=0.080" \
  "++render_export_close_kernel=1" \
  "++render_export_erode_kernel=1" \
  "++render_export_mask_source=$MASK_SOURCE" \
  "++render_export_mask_erode_kernel=$MASK_ERODE_KERNEL" \
  "++render_export_fill_close_kernel=1" \
  "++render_export_clip_to_mask=true" \
  "++render_export_fill_dilate_kernel=1" \
  "export_semantic_editable_assets=true" \
  "semantic_editable_parser_root=$PARSER_ROOT" \
  "semantic_editable_parser_layout=cihp_subject" \
  "semantic_editable_direct_parser_mode=true" \
  "semantic_editable_export_compact_head=true" \
  "semantic_editable_include_binding_summary=true" \
  "+semantic_editable_preview_min_area=18" \
  "hydra.run.dir=$LOG_DIR/hydra_runtime" \
  "wandb_disable=true" 2>&1 | tee "$LOG_DIR/render.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/analyze_377_head_silhouette_artifact.py \
  --render-exp "$OUT_EXP" \
  --baseline-render-exp "$BASE_RENDER" \
  --dataset-root "$DATA_ROOT" \
  --subject CoreView_377 \
  --split test-view \
  --select "${HEAD_SELECT[@]}" \
  --out-dir "$OUT_EXP/diagnostics/head_silhouette_rgbclip" \
  --panels gt gt_mask base_render render outside_gt base_diff layer region compact_semantic semantic \
  2>&1 | tee "$LOG_DIR/head_silhouette.log"

"$PYTHON_BIN" tools/make_377_render_comparison_montage.py \
  --render-exp "$BASE_RENDER" "$OUT_EXP" \
  --labels base rgbclip \
  --gt-root "$DATA_ROOT/CoreView_377" \
  --split test-view \
  --output-dir "$LOG_DIR/compare_headcrop" \
  --select "${HEAD_SELECT[@]}" \
  --panel-width 230 \
  --header-height 34 \
  --stack 2>&1 | tee "$LOG_DIR/montage.log"

echo "OUT_EXP=$OUT_EXP"
echo "LOG_DIR=$LOG_DIR"
echo "METRICS=$OUT_EXP/diagnostics/head_silhouette_rgbclip/head_silhouette_metrics.tsv"
echo "MONTAGE=$LOG_DIR/compare_headcrop/stacked_comparison.png"
