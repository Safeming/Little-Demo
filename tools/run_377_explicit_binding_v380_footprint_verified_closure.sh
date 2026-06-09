#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
RUN_ID="${RUN_ID:-v380_footprint_verified_closure_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v380_footprint_verified_closure_${RUN_ID}}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v380_footprint_verified_closure_${RUN_ID}}"
ASSET_DIR="$LOG_DIR/assets"
EVENTS="$LOG_DIR/events.tsv"
ASSET_JSON="$ASSET_DIR/v380_footprint_verified_closure_asset.json"
ASSET_TSV="$ASSET_DIR/v380_footprint_verified_closure_actions.tsv"

BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
SEED_ASSET_JSON="${SEED_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v373_quota_coverage_grouped_actuator_v373b_quota_rr_full_20260527_152104_bjt/assets/v371_seed_residual_component_multi_micro_grouped_actuator_asset.json}"
BASE_ASSET_JSON="${BASE_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v374_portfolio_merge_grouped_actuator_v374_v374_v376_queue_20260527_192801_bjt/assets/v374_portfolio_merge_grouped_actuator_asset.json}"
VALIDATION_TSV="${VALIDATION_TSV:-$ROOT/exp/stageB/logs/377_explicit_binding_v373_quota_coverage_grouped_actuator_v373b_quota_rr_full_20260527_152104_bjt/group_validation.tsv}"
MAX_ACTIONS="${MAX_ACTIONS:-144}"
VISIBLE_RADIUS_PX="${VISIBLE_RADIUS_PX:-42}"
ANCHOR_IDS="${ANCHOR_IDS:-8}"
FOOTPRINT_RADIUS_PX="${FOOTPRINT_RADIUS_PX:-6}"
MIN_FOOTPRINT_INNER_PIXELS="${MIN_FOOTPRINT_INNER_PIXELS:-1}"
MAX_FOOTPRINT_OUTER_PIXELS="${MAX_FOOTPRINT_OUTER_PIXELS:-0}"
CHILD_OPACITY="${CHILD_OPACITY:-0.04}"
TRAIN_ON_STRICT_PASS="${TRAIN_ON_STRICT_PASS:-true}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"

RAW_GATE_RUN_ID="formal_377_v380_footprint_verified_closure_raw_gate_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')"
RAW_GATE_LOG="$LOG_DIR/v380_raw_gate.launcher.log"
RAW_GATE_SUMMARY="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${RAW_GATE_RUN_ID}/summary.tsv"
TRAIN_LAUNCH_LOG="$LOG_DIR/v380_semantic_train.launcher.log"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$DATA_ROOT" "$CANDIDATE_CKPT" "$SEED_ASSET_JSON" "$BASE_ASSET_JSON" "$VALIDATION_TSV"; do
  [ -e "$required" ] || { echo "missing required path: $required" >&2; exit 2; }
done
mkdir -p "$LOG_DIR" "$EXP_ROOT" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
log_event(){ printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"; }

log_event "make_asset_start" "$ASSET_JSON"
env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" tools/make_377_stageB_v378_visible_contributor_closure_asset.py \
  --config-path "$BASE_EXP/.hydra/config.yaml" --checkpoint "$CANDIDATE_CKPT" --dataset-root "$DATA_ROOT" --subject CoreView_377 \
  --seed-json "$SEED_ASSET_JSON" --base-json "$BASE_ASSET_JSON" --validation-tsv "$VALIDATION_TSV" --out-json "$ASSET_JSON" --out-tsv "$ASSET_TSV" \
  --max-actions "$MAX_ACTIONS" --visible-radius-px "$VISIBLE_RADIUS_PX" --anchor-ids "$ANCHOR_IDS" \
  --footprint-radius-px "$FOOTPRINT_RADIUS_PX" --min-footprint-inner-pixels "$MIN_FOOTPRINT_INNER_PIXELS" --max-footprint-outer-pixels "$MAX_FOOTPRINT_OUTER_PIXELS" \
  > "$LOG_DIR/make_v380_footprint_verified_closure_asset.log" 2>&1
log_event "make_asset_done" "$ASSET_JSON"

log_event "raw_gate_start" "$RAW_GATE_RUN_ID"
env -u LOG_DIR -u EXP_ROOT -u HYDRA_RUN_ROOT GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  RUN_ID="$RAW_GATE_RUN_ID" CANDIDATE_CKPT="$CANDIDATE_CKPT" CANDIDATE_VARIANT_NAME="candidate_v380_footprint_verified_closure" \
  CANDIDATE_SPLIT_CHILD_COMPONENT_ENABLE=true CANDIDATE_SPLIT_CHILD_COMPONENT_ASSET_JSON="$ASSET_JSON" \
  CANDIDATE_SPLIT_CHILD_COMPONENT_ACTION_REQUIRED=false CANDIDATE_SPLIT_CHILD_COMPONENT_OPACITY="$CHILD_OPACITY" \
  CANDIDATE_SPLIT_CHILD_COMPONENT_RADIUS_SCALE=1.0 CANDIDATE_SPLIT_CHILD_COMPONENT_MAX_CHILDREN=-1 \
  "$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh" > "$RAW_GATE_LOG" 2>&1
RAW_GATE_STATUS="$("$PYTHON_BIN" - "$RAW_GATE_SUMMARY" <<'PY'
import csv,sys
s='missing'
with open(sys.argv[1],encoding='utf-8',newline='') as h:
    for r in csv.DictReader(h,delimiter='\t'):
        if r.get('variant')=='candidate_v380_footprint_verified_closure': s=r.get('status','')
print(s)
PY
)"
log_event "raw_gate_done" "status=$RAW_GATE_STATUS summary=$RAW_GATE_SUMMARY"

TRAIN_PID=""; TRAIN_EXP_DIR=""; TRAIN_EST_END_BJT=""
if [ "$TRAIN_ON_STRICT_PASS" = "true" ] && [ "$RAW_GATE_STATUS" = "strict_pass" ]; then
  TRAIN_RUN_ID="formal_377_v380_footprint_verified_closure_semantic_train_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')"
  TRAIN_EXP_DIR="$ROOT/exp/formal/377_v380_footprint_verified_closure_semantic_train_${TRAIN_RUN_ID}"
  TRAIN_SCRIPT="$LOG_DIR/v380_semantic_train.launch.sh"
  cat > "$TRAIN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" BASE_CKPT="$CANDIDATE_CKPT" RUN_ID="$TRAIN_RUN_ID" EXP_DIR="$TRAIN_EXP_DIR" TRAIN_STEPS="$TRAIN_STEPS" \\
"$ROOT/tools/formal/run_377_v338_semantic_train.sh" \\
  "++pipeline.split_child_component_enable=true" "++pipeline.split_child_component_asset_json=$ASSET_JSON" \\
  "++pipeline.split_child_component_action_required=false" "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \\
  "++pipeline.split_child_component_radius_scale=1.0" "++pipeline.split_child_component_max_children=-1"
EOF
  chmod +x "$TRAIN_SCRIPT"; log_event "train_start" "$TRAIN_EXP_DIR"; "$TRAIN_SCRIPT" > "$TRAIN_LAUNCH_LOG" 2>&1 &
  TRAIN_PID="$!"; echo "$TRAIN_PID" > "$LOG_DIR/train.pid"; TRAIN_EST_END_BJT="$(TZ=Asia/Shanghai date -d '+65 minutes' '+%F %T BJT')"; log_event "train_launched" "pid=$TRAIN_PID est_end=$TRAIN_EST_END_BJT"
fi
END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"; log_event "finished_bjt" "$END_BJT"
echo "LOG_DIR=$LOG_DIR"; echo "ASSET_JSON=$ASSET_JSON"; echo "RAW_GATE_STATUS=$RAW_GATE_STATUS"; echo "RAW_GATE_SUMMARY=$RAW_GATE_SUMMARY"; echo "TRAIN_PID=$TRAIN_PID"; echo "TRAIN_EXP_DIR=$TRAIN_EXP_DIR"; echo "TRAIN_EST_END_BJT=$TRAIN_EST_END_BJT"; echo "END_BJT=$END_BJT"
