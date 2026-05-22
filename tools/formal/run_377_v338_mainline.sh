#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
RUN_ID="${RUN_ID:-formal_377_v338_mainline_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/formal/logs/377_v338_mainline_${RUN_ID}}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"

TRAIN_EXP_DIR="${TRAIN_EXP_DIR:-$ROOT/exp/formal/377_v338_semantic_train_${RUN_ID}}"
GATE_RUN_ID="${GATE_RUN_ID:-${RUN_ID}_gate}"
EXPORT_RUN_ID="${EXPORT_RUN_ID:-${RUN_ID}_export}"
EXPORT_EXP_DIR="${EXPORT_EXP_DIR:-$ROOT/exp/formal/377_v338_semantic_assets_${RUN_ID}}"

mkdir -p "$LOG_DIR"
EVENTS="$LOG_DIR/events.tsv"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

find_latest_ckpt() {
  "$PYTHON_BIN" - "$1" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
ckpts = []
for path in root.glob("ckpt*.pth"):
    m = re.match(r"ckpt(\d+)\.pth$", path.name)
    if m:
        ckpts.append((int(m.group(1)), path))
if not ckpts:
    raise SystemExit(f"no checkpoint found in {root}")
print(max(ckpts)[1])
PY
}

check_gate_status() {
  "$PYTHON_BIN" - "$1" <<'PY'
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
if not summary.exists():
    raise SystemExit(f"summary missing: {summary}")
with summary.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
candidate = [row for row in rows if row.get("variant") == "candidate_v338_temporal_selector_grow_only_guard"]
if not candidate:
    raise SystemExit("candidate row missing from gate summary")
row = candidate[-1]
status = row.get("status", "")
print(status)
if status != "strict_pass":
    raise SystemExit(f"candidate gate failed: status={status}")
PY
}

log_event train_start "$TRAIN_EXP_DIR"
TRAIN_STEPS="$TRAIN_STEPS" \
RUN_ID="$RUN_ID" \
EXP_DIR="$TRAIN_EXP_DIR" \
"$ROOT/tools/formal/run_377_v338_semantic_train.sh" "$@" \
  2>&1 | tee "$LOG_DIR/train.log"
log_event train_done "$TRAIN_EXP_DIR"

CANDIDATE_CKPT="$(find_latest_ckpt "$TRAIN_EXP_DIR")"
echo "CANDIDATE_CKPT=$CANDIDATE_CKPT" | tee "$LOG_DIR/candidate_ckpt.env"

log_event gate_start "$GATE_RUN_ID"
CANDIDATE_CKPT="$CANDIDATE_CKPT" \
RUN_ID="$GATE_RUN_ID" \
"$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh" \
  2>&1 | tee "$LOG_DIR/gate.log"

GATE_SUMMARY="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${GATE_RUN_ID}/summary.tsv"
GATE_STATUS="$(check_gate_status "$GATE_SUMMARY")"
log_event gate_done "status=$GATE_STATUS summary=$GATE_SUMMARY"

log_event export_start "$EXPORT_EXP_DIR"
BASE_CKPT="$CANDIDATE_CKPT" \
RUN_ID="$EXPORT_RUN_ID" \
EXP_DIR="$EXPORT_EXP_DIR" \
EXPORT_INTERPRETABILITY=true \
EXPORT_EDITABLE=true \
EXPORT_OPACITY=false \
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}" \
COMPACT_MAPPING_FILE="${COMPACT_MAPPING_FILE:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}" \
"$ROOT/tools/formal/run_377_signed_geometry_export.sh" \
  "dataset.parsing_prior.enable=true" \
  "dataset.parsing_prior.roi_enable=false" \
  "dataset.parsing_prior.parser_root=${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}" \
  "dataset.parsing_prior.parser_layout=cihp_subject" \
  "dataset.parsing_prior.use_direct_parser_labels=true" \
  "dataset.parsing_prior.compact_mapping_file=${COMPACT_MAPPING_FILE:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}" \
  "++semantic_editable_use_direct_parser=true" \
  "++semantic_editable_export_compact_head=true" \
  2>&1 | tee "$LOG_DIR/export.log"

ASSET_ROOT="$EXPORT_EXP_DIR/test-view/semantic_editable_assets"
"$PYTHON_BIN" tools/check_semantic_editable_assets.py \
  --asset-root "$ASSET_ROOT" \
  --min-views 1 \
  --require-compact \
  --require-parser-preview \
  | tee "$LOG_DIR/asset_check.json"
log_event export_done "$ASSET_ROOT"

cat > "$LOG_DIR/mainline_result.env" <<EOF
RUN_ID=$RUN_ID
TRAIN_EXP_DIR=$TRAIN_EXP_DIR
CANDIDATE_CKPT=$CANDIDATE_CKPT
GATE_SUMMARY=$GATE_SUMMARY
EXPORT_EXP_DIR=$EXPORT_EXP_DIR
ASSET_ROOT=$ASSET_ROOT
EOF

echo "MAINLINE_RESULT=$LOG_DIR/mainline_result.env"
