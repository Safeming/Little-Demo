#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
RUN_ID="${RUN_ID:-v374_portfolio_merge_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v374_portfolio_merge_grouped_actuator_${RUN_ID}}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v374_portfolio_merge_grouped_actuator_${RUN_ID}}"
ASSET_DIR="$LOG_DIR/assets"
EVENTS="$LOG_DIR/events.tsv"
MERGE_SUMMARY="$LOG_DIR/summary.tsv"
ASSET_JSON="$ASSET_DIR/v374_portfolio_merge_grouped_actuator_asset.json"

CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
BASE_ASSET_JSON="${BASE_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v371_strength_sweep_grouped_actuator_v371_strength_sweep_full_20260526_172001_bjt/assets/v371_validated_residual_component_multi_micro_grouped_actuator_asset.json}"
ADDON_ASSET_JSON="${ADDON_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v373_quota_coverage_grouped_actuator_v373b_quota_rr_full_20260527_152104_bjt/assets/v371_validated_residual_component_multi_micro_grouped_actuator_asset.json}"
ADDON_VALIDATION_TSV="${ADDON_VALIDATION_TSV:-$ROOT/exp/stageB/logs/377_explicit_binding_v373_quota_coverage_grouped_actuator_v373b_quota_rr_full_20260527_152104_bjt/group_validation.tsv}"
ADDON_VIEWS="${ADDON_VIEWS:-c22}"
MAX_ADDON_PER_VIEW="${MAX_ADDON_PER_VIEW:-4}"
CHILD_OPACITY="${CHILD_OPACITY:-0.04}"
RAW_GATE_ENABLE="${RAW_GATE_ENABLE:-true}"
TRAIN_ON_STRICT_PASS="${TRAIN_ON_STRICT_PASS:-true}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"

RAW_GATE_RUN_ID="formal_377_v374_portfolio_merge_grouped_actuator_raw_gate_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')"
RAW_GATE_LOG="$LOG_DIR/v374_raw_gate.launcher.log"
RAW_GATE_SUMMARY="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${RAW_GATE_RUN_ID}/summary.tsv"
TRAIN_LAUNCH_LOG="$LOG_DIR/v374_semantic_train.launcher.log"

for required in "$PYTHON_BIN" "$CANDIDATE_CKPT" "$BASE_ASSET_JSON" "$ADDON_ASSET_JSON" "$ADDON_VALIDATION_TSV"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR" "$EXP_ROOT" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

log_event "merge_start" "$ASSET_JSON"
"$PYTHON_BIN" tools/merge_377_stageB_v374_portfolio_asset.py \
  --base-json "$BASE_ASSET_JSON" \
  --addon-json "$ADDON_ASSET_JSON" \
  --addon-validation-tsv "$ADDON_VALIDATION_TSV" \
  --addon-views "$ADDON_VIEWS" \
  --max-addon-per-view "$MAX_ADDON_PER_VIEW" \
  --out-json "$ASSET_JSON" \
  --summary-tsv "$MERGE_SUMMARY" \
  > "$LOG_DIR/merge_v374_portfolio_asset.log" 2>&1
log_event "merge_done" "$ASSET_JSON"

RAW_GATE_STATUS="skip"
if [ "$RAW_GATE_ENABLE" = "true" ]; then
  log_event "raw_gate_start" "$RAW_GATE_RUN_ID"
  env -u LOG_DIR -u EXP_ROOT -u HYDRA_RUN_ROOT \
    GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
    RUN_ID="$RAW_GATE_RUN_ID" \
    CANDIDATE_CKPT="$CANDIDATE_CKPT" \
    CANDIDATE_VARIANT_NAME="candidate_v374_portfolio_merge_grouped_actuator" \
    CANDIDATE_SPLIT_CHILD_COMPONENT_ENABLE=true \
    CANDIDATE_SPLIT_CHILD_COMPONENT_ASSET_JSON="$ASSET_JSON" \
    CANDIDATE_SPLIT_CHILD_COMPONENT_ACTION_REQUIRED=false \
    CANDIDATE_SPLIT_CHILD_COMPONENT_OPACITY="$CHILD_OPACITY" \
    CANDIDATE_SPLIT_CHILD_COMPONENT_RADIUS_SCALE=1.0 \
    CANDIDATE_SPLIT_CHILD_COMPONENT_MAX_CHILDREN=-1 \
    "$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh" \
    > "$RAW_GATE_LOG" 2>&1
  RAW_GATE_STATUS="$("$PYTHON_BIN" - "$RAW_GATE_SUMMARY" <<'PY'
import csv, sys
status = "missing"
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row.get("variant") == "candidate_v374_portfolio_merge_grouped_actuator":
            status = row.get("status", "")
print(status)
PY
)"
  log_event "raw_gate_done" "status=$RAW_GATE_STATUS summary=$RAW_GATE_SUMMARY"
fi

TRAIN_PID=""
TRAIN_EXP_DIR=""
TRAIN_EST_END_BJT=""
if [ "$TRAIN_ON_STRICT_PASS" = "true" ] && [ "$RAW_GATE_STATUS" = "strict_pass" ]; then
  TRAIN_RUN_ID="formal_377_v374_portfolio_merge_grouped_actuator_semantic_train_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')"
  TRAIN_EXP_DIR="$ROOT/exp/formal/377_v374_portfolio_merge_grouped_actuator_semantic_train_${TRAIN_RUN_ID}"
  TRAIN_SCRIPT="$LOG_DIR/v374_semantic_train.launch.sh"
  cat > "$TRAIN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \\
BASE_CKPT="$CANDIDATE_CKPT" RUN_ID="$TRAIN_RUN_ID" EXP_DIR="$TRAIN_EXP_DIR" TRAIN_STEPS="$TRAIN_STEPS" \\
"$ROOT/tools/formal/run_377_v338_semantic_train.sh" \\
  "++pipeline.split_child_component_enable=true" \\
  "++pipeline.split_child_component_asset_json=$ASSET_JSON" \\
  "++pipeline.split_child_component_action_required=false" \\
  "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \\
  "++pipeline.split_child_component_radius_scale=1.0" \\
  "++pipeline.split_child_component_max_children=-1"
EOF
  chmod +x "$TRAIN_SCRIPT"
  log_event "train_start" "$TRAIN_EXP_DIR"
  "$TRAIN_SCRIPT" > "$TRAIN_LAUNCH_LOG" 2>&1 &
  TRAIN_PID="$!"
  echo "$TRAIN_PID" > "$LOG_DIR/train.pid"
  TRAIN_EST_END_BJT="$(TZ=Asia/Shanghai date -d '+65 minutes' '+%F %T BJT')"
  log_event "train_launched" "pid=$TRAIN_PID est_end=$TRAIN_EST_END_BJT log=$TRAIN_LAUNCH_LOG"
fi

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "finished_bjt" "$END_BJT"
echo "LOG_DIR=$LOG_DIR"
echo "ASSET_JSON=$ASSET_JSON"
echo "RAW_GATE_STATUS=$RAW_GATE_STATUS"
echo "RAW_GATE_SUMMARY=$RAW_GATE_SUMMARY"
echo "TRAIN_PID=$TRAIN_PID"
echo "TRAIN_EXP_DIR=$TRAIN_EXP_DIR"
echo "TRAIN_EST_END_BJT=$TRAIN_EST_END_BJT"
echo "END_BJT=$END_BJT"
