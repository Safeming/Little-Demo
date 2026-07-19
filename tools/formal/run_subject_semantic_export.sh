#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
SUBJECT="${SUBJECT:?set SUBJECT}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING_FILE="${COMPACT_MAPPING_FILE:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:?set BASE_EXP}"
BASE_CKPT="${BASE_CKPT:?set BASE_CKPT}"
EXP_DIR="${EXP_DIR:?set EXP_DIR}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"
TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:?set TEST_VIEWS_SPEC}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:?set TEST_FRAMES_SPEC}"
EXPORT_INTERPRETABILITY="${EXPORT_INTERPRETABILITY:-true}"
EXPORT_EDITABLE="${EXPORT_EDITABLE:-true}"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" \
  "$DATA_ROOT/$SUBJECT" "$PARSER_ROOT/$SUBJECT/mask_cihp" "$COMPACT_MAPPING_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_DIR" "$HYDRA_RUN_DIR"

env \
  "CUDA_VISIBLE_DEVICES=$GPU" \
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB" \
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB" \
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB" \
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB" \
  "PYTHONUNBUFFERED=1" \
  "$PYTHON_BIN" render.py \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  mode=test \
  "load_ckpt=$BASE_CKPT" \
  "exp_dir=$EXP_DIR" \
  "dataset.root_dir=$DATA_ROOT" \
  "dataset.preload=false" \
  "dataset.subject=$SUBJECT" \
  "dataset.train_views=$TRAIN_VIEWS_SPEC" \
  "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
  "dataset.test_views.view=$TEST_VIEWS_SPEC" \
  "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
  "dataset.parsing_prior.enable=true" \
  "dataset.parsing_prior.roi_enable=false" \
  "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
  "dataset.parsing_prior.parser_layout=cihp_subject" \
  "dataset.parsing_prior.use_direct_parser_labels=true" \
  "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING_FILE" \
  "++semantic_editable_use_direct_parser=true" \
  "++semantic_editable_export_compact_head=true" \
  "export_interpretability=$EXPORT_INTERPRETABILITY" \
  "export_semantic_editable_assets=$EXPORT_EDITABLE" \
  "++render_export_refine=false" \
  "++resume.partial_converter_missing_keys_allow_patterns=[texture.structured_trunk_,camera_geometry.,texture.shadow_handoff_approved]" \
  "hydra.run.dir=$HYDRA_RUN_DIR" \
  "wandb_disable=true" \
  "$@"

echo "SEMANTIC_EXPORT_EXP=$EXP_DIR/test-view"
