#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

RUN_ID="${RUN_ID:-v393_v392_mainline_gate_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/formal/logs/377_v393_v392_mainline_gate_${RUN_ID}}"
GATE_RUN_ID="${GATE_RUN_ID:-formal_377_v393_v392_dense_gate_${RUN_ID}}"
EXPORT_RUN_ID="${EXPORT_RUN_ID:-formal_377_v393_v392_semantic_export_${RUN_ID}}"
EXPORT_EXP_DIR="${EXPORT_EXP_DIR:-$ROOT/exp/formal/377_v393_v392_semantic_export_${RUN_ID}}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v392_legacy_stacked_directional_residual_v392_legacy_stacked_directional_residual_20260531_125104_bjt/ckpt140160.pth}"
ASSET_JSON="${ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v387_runtime_bounded_marginal_selector_v387_runtime_bounded_marginal_selector_20260530_094841_bjt/v387_selector/assets/v387_final_runtime_bounded_marginal_selector_asset.json}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING_FILE="${COMPACT_MAPPING_FILE:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
DENSE_TEST_VIEWS_SPEC="${DENSE_TEST_VIEWS_SPEC:-[21,22,23]}"
DENSE_TEST_FRAMES_SPEC="${DENSE_TEST_FRAMES_SPEC:-[0,570,1]}"
EXPORT_TEST_VIEWS_SPEC="${EXPORT_TEST_VIEWS_SPEC:-[21,22,23]}"
EXPORT_TEST_FRAMES_SPEC="${EXPORT_TEST_FRAMES_SPEC:-[0,570,60]}"
RENDER_EXPORT_OPACITY_THRESHOLD="${RENDER_EXPORT_OPACITY_THRESHOLD:-0.06}"

EVENTS="$LOG_DIR/events.tsv"
RUN_INFO="$LOG_DIR/run_info.txt"
DENSE_GATE_LOG="$LOG_DIR/dense_gate.log"
EXPORT_LOG="$LOG_DIR/semantic_export.log"
ASSET_CHECK_JSON="$LOG_DIR/asset_check.json"
RESULT_JSON="$LOG_DIR/result.json"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$CANDIDATE_CKPT" \
  "$ASSET_JSON" "$DATA_ROOT" "$PARSER_ROOT/CoreView_377" "$COMPACT_MAPPING_FILE" \
  "$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh" \
  "$ROOT/tools/formal/run_377_signed_geometry_export.sh" \
  "$ROOT/tools/check_semantic_editable_assets.py"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR" "$EXPORT_EXP_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d '+260 minutes' '+%F %T BJT')"
cat > "$RUN_INFO" <<INFO
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
CANDIDATE_CKPT=$CANDIDATE_CKPT
ASSET_JSON=$ASSET_JSON
LOG_DIR=$LOG_DIR
GATE_RUN_ID=$GATE_RUN_ID
EXPORT_RUN_ID=$EXPORT_RUN_ID
EXPORT_EXP_DIR=$EXPORT_EXP_DIR
DENSE_TEST_VIEWS_SPEC=$DENSE_TEST_VIEWS_SPEC
DENSE_TEST_FRAMES_SPEC=$DENSE_TEST_FRAMES_SPEC
EXPORT_TEST_VIEWS_SPEC=$EXPORT_TEST_VIEWS_SPEC
EXPORT_TEST_FRAMES_SPEC=$EXPORT_TEST_FRAMES_SPEC
CONFIG_CONTRACT=v392_selected_mainline_dense_raw_gate_then_semantic_editable_export_check
INFO

log_event dense_gate_start "$GATE_RUN_ID"
CANDIDATE_CKPT="$CANDIDATE_CKPT" \
CANDIDATE_VARIANT_NAME="candidate_v393_v392_selected_dense" \
CANDIDATE_SPLIT_CHILD_COMPONENT_ENABLE=true \
CANDIDATE_SPLIT_CHILD_COMPONENT_ASSET_JSON="$ASSET_JSON" \
CANDIDATE_SPLIT_CHILD_COMPONENT_ACTION_REQUIRED=false \
CANDIDATE_SPLIT_CHILD_COMPONENT_OPACITY=0.045 \
CANDIDATE_SPLIT_CHILD_COMPONENT_RADIUS_SCALE=1.0 \
CANDIDATE_SPLIT_CHILD_COMPONENT_MAX_CHILDREN=-1 \
RUN_ID="$GATE_RUN_ID" \
GPU="$GPU" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
DATA_ROOT="$DATA_ROOT" \
BASE_EXP="$BASE_EXP" \
BASE_CKPT="$BASE_CKPT" \
TRAIN_VIEWS_SPEC="$TRAIN_VIEWS_SPEC" \
TRAIN_FRAMES_SPEC="$TRAIN_FRAMES_SPEC" \
TEST_VIEWS_SPEC="$DENSE_TEST_VIEWS_SPEC" \
TEST_FRAMES_SPEC="$DENSE_TEST_FRAMES_SPEC" \
RENDER_EXPORT_OPACITY_THRESHOLD="$RENDER_EXPORT_OPACITY_THRESHOLD" \
"$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh" \
  > "$DENSE_GATE_LOG" 2>&1

GATE_SUMMARY="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${GATE_RUN_ID}/summary.tsv"
GATE_WORST="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${GATE_RUN_ID}/worst_frames.tsv"
log_event dense_gate_done "$GATE_SUMMARY"

log_event semantic_export_start "$EXPORT_EXP_DIR"
BASE_CKPT="$CANDIDATE_CKPT" \
RUN_ID="$EXPORT_RUN_ID" \
EXP_DIR="$EXPORT_EXP_DIR" \
GPU="$GPU" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
DATA_ROOT="$DATA_ROOT" \
BASE_EXP="$BASE_EXP" \
TRAIN_VIEWS_SPEC="$TRAIN_VIEWS_SPEC" \
TRAIN_FRAMES_SPEC="$TRAIN_FRAMES_SPEC" \
TEST_VIEWS_SPEC="$EXPORT_TEST_VIEWS_SPEC" \
TEST_FRAMES_SPEC="$EXPORT_TEST_FRAMES_SPEC" \
EXPORT_INTERPRETABILITY=true \
EXPORT_EDITABLE=true \
EXPORT_OPACITY=false \
FORMAL_PRESET=v338_temporal_selector_grow_only_guard \
"$ROOT/tools/formal/run_377_signed_geometry_export.sh" \
  "dataset.parsing_prior.enable=true" \
  "dataset.parsing_prior.roi_enable=false" \
  "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
  "dataset.parsing_prior.parser_layout=cihp_subject" \
  "dataset.parsing_prior.use_direct_parser_labels=true" \
  "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING_FILE" \
  "++semantic_editable_use_direct_parser=true" \
  "++semantic_editable_export_compact_head=true" \
  "++pipeline.split_child_component_enable=true" \
  "++pipeline.split_child_component_asset_json=$ASSET_JSON" \
  "++pipeline.split_child_component_action_required=false" \
  "++pipeline.split_child_component_opacity=0.045" \
  "++pipeline.split_child_component_radius_scale=1.0" \
  "++pipeline.split_child_component_max_children=-1" \
  > "$EXPORT_LOG" 2>&1

ASSET_ROOT="$EXPORT_EXP_DIR/test-view/semantic_editable_assets"
"$PYTHON_BIN" tools/check_semantic_editable_assets.py \
  --asset-root "$ASSET_ROOT" \
  --min-views 1 \
  --require-compact \
  --require-parser-preview \
  > "$ASSET_CHECK_JSON"
log_event semantic_export_done "$ASSET_ROOT"

"$PYTHON_BIN" - "$RESULT_JSON" "$GATE_SUMMARY" "$GATE_WORST" "$ASSET_CHECK_JSON" "$ASSET_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
gate_summary = Path(sys.argv[2])
gate_worst = Path(sys.argv[3])
asset_check = Path(sys.argv[4])
asset_root = Path(sys.argv[5])

rows = []
with gate_summary.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
candidate = next((row for row in rows if row.get("variant") == "candidate_v393_v392_selected_dense"), None)
asset_payload = json.loads(asset_check.read_text(encoding="utf-8"))
payload = {
    "gate_summary": str(gate_summary),
    "gate_worst": str(gate_worst),
    "candidate": candidate,
    "asset_check": asset_payload,
    "asset_root": str(asset_root),
}
result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
if candidate is None:
    raise SystemExit("candidate row missing")
if asset_payload.get("status") != "pass":
    raise SystemExit("asset check failed")
PY

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "END_BJT=$END_BJT" >> "$RUN_INFO"
log_event all_done "$RESULT_JSON"

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "GATE_SUMMARY=$GATE_SUMMARY"
echo "GATE_WORST=$GATE_WORST"
echo "EXPORT_EXP_DIR=$EXPORT_EXP_DIR"
echo "ASSET_ROOT=$ASSET_ROOT"
echo "RESULT_JSON=$RESULT_JSON"
echo "START_BJT=$START_BJT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
