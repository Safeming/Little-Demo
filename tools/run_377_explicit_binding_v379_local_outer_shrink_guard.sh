#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
RUN_ID="${RUN_ID:-v379_local_outer_shrink_guard_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v379_local_outer_shrink_guard_${RUN_ID}}"
ASSET_DIR="$LOG_DIR/assets"
EVENTS="$LOG_DIR/events.tsv"
POINT_PRIOR_JSON="$ASSET_DIR/v379_local_outer_shrink_prior.json"

CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
ASSET_JSON="${ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v374_portfolio_merge_grouped_actuator_v374_v374_v376_queue_20260527_192801_bjt/assets/v374_portfolio_merge_grouped_actuator_asset.json}"
WORST_FRAMES_TSV="${WORST_FRAMES_TSV:-$ROOT/exp/formal/logs/377_v338_raw_contour_gate_formal_377_v374_trained_ckpt_raw_gate_20260528_094800_bjt/worst_frames.tsv}"
MAX_IMAGES="${MAX_IMAGES:-12}"
MAX_POINTS="${MAX_POINTS:-96}"
SHRINK_FACTOR="${SHRINK_FACTOR:-0.94}"
CHILD_OPACITY="${CHILD_OPACITY:-0.04}"
TRAIN_ON_STRICT_PASS="${TRAIN_ON_STRICT_PASS:-true}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"

RAW_GATE_RUN_ID="formal_377_v379_local_outer_shrink_guard_raw_gate_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')"
RAW_GATE_LOG="$LOG_DIR/v379_raw_gate.launcher.log"
RAW_GATE_SUMMARY="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${RAW_GATE_RUN_ID}/summary.tsv"
TRAIN_LAUNCH_LOG="$LOG_DIR/v379_semantic_train.launcher.log"

for required in "$PYTHON_BIN" "$CANDIDATE_CKPT" "$ASSET_JSON" "$WORST_FRAMES_TSV"; do
  [ -e "$required" ] || { echo "missing required path: $required" >&2; exit 2; }
done
mkdir -p "$LOG_DIR" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
log_event(){ printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"; }

log_event "make_prior_start" "$POINT_PRIOR_JSON"
"$PYTHON_BIN" tools/make_377_stageB_v379_local_outer_shrink_prior.py \
  --asset-json "$ASSET_JSON" --worst-frames-tsv "$WORST_FRAMES_TSV" --out-json "$POINT_PRIOR_JSON" \
  --max-images "$MAX_IMAGES" --max-points "$MAX_POINTS" > "$LOG_DIR/make_v379_local_outer_shrink_prior.log" 2>&1
log_event "make_prior_done" "$POINT_PRIOR_JSON"

log_event "raw_gate_start" "$RAW_GATE_RUN_ID"
env -u LOG_DIR -u EXP_ROOT -u HYDRA_RUN_ROOT GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  RUN_ID="$RAW_GATE_RUN_ID" CANDIDATE_CKPT="$CANDIDATE_CKPT" CANDIDATE_VARIANT_NAME="candidate_v379_local_outer_shrink_guard" \
  CANDIDATE_SPLIT_CHILD_COMPONENT_ENABLE=true CANDIDATE_SPLIT_CHILD_COMPONENT_ASSET_JSON="$ASSET_JSON" \
  CANDIDATE_SPLIT_CHILD_COMPONENT_ACTION_REQUIRED=false CANDIDATE_SPLIT_CHILD_COMPONENT_OPACITY="$CHILD_OPACITY" \
  CANDIDATE_SPLIT_CHILD_COMPONENT_RADIUS_SCALE=1.0 CANDIDATE_SPLIT_CHILD_COMPONENT_MAX_CHILDREN=-1 \
  CANDIDATE_SIGNED_POINT_JSON="$POINT_PRIOR_JSON" CANDIDATE_SIGNED_POINT_SCREEN_ACTUATOR_ENABLE=true \
  CANDIDATE_SIGNED_SHRINK_FACTOR=1.0 CANDIDATE_SIGNED_GROW_FACTOR=1.0 CANDIDATE_SIGNED_MAX_SHRINK_POINTS="$MAX_POINTS" CANDIDATE_SIGNED_MAX_GROW_POINTS=0 \
  CANDIDATE_SIGNED_SCREEN_ACTUATOR_ENABLE=true CANDIDATE_SIGNED_SCREEN_NORMAL_SHRINK_FACTOR="$SHRINK_FACTOR" CANDIDATE_SIGNED_SCREEN_NORMAL_GROW_FACTOR=1.0 CANDIDATE_SIGNED_SCREEN_TANGENT_FACTOR=1.0 \
  CANDIDATE_SIGNED_CENTER_OFFSET_ENABLE=true CANDIDATE_SIGNED_CENTER_OFFSET_OUTER_PX=0.35 CANDIDATE_SIGNED_CENTER_OFFSET_INNER_PX=0.0 \
  "$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh" > "$RAW_GATE_LOG" 2>&1
RAW_GATE_STATUS="$("$PYTHON_BIN" - "$RAW_GATE_SUMMARY" <<'PY'
import csv,sys
s='missing'
with open(sys.argv[1],encoding='utf-8',newline='') as h:
    for r in csv.DictReader(h,delimiter='\t'):
        if r.get('variant')=='candidate_v379_local_outer_shrink_guard': s=r.get('status','')
print(s)
PY
)"
log_event "raw_gate_done" "status=$RAW_GATE_STATUS summary=$RAW_GATE_SUMMARY"

TRAIN_PID=""; TRAIN_EXP_DIR=""; TRAIN_EST_END_BJT=""
if [ "$TRAIN_ON_STRICT_PASS" = "true" ] && [ "$RAW_GATE_STATUS" = "strict_pass" ]; then
  TRAIN_RUN_ID="formal_377_v379_local_outer_shrink_guard_semantic_train_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')"
  TRAIN_EXP_DIR="$ROOT/exp/formal/377_v379_local_outer_shrink_guard_semantic_train_${TRAIN_RUN_ID}"
  TRAIN_SCRIPT="$LOG_DIR/v379_semantic_train.launch.sh"
  cat > "$TRAIN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" BASE_CKPT="$CANDIDATE_CKPT" RUN_ID="$TRAIN_RUN_ID" EXP_DIR="$TRAIN_EXP_DIR" TRAIN_STEPS="$TRAIN_STEPS" \\
"$ROOT/tools/formal/run_377_v338_semantic_train.sh" \\
  "++pipeline.split_child_component_enable=true" "++pipeline.split_child_component_asset_json=$ASSET_JSON" \\
  "++pipeline.split_child_component_action_required=false" "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \\
  "++pipeline.split_child_component_radius_scale=1.0" "++pipeline.split_child_component_max_children=-1" \\
  "pipeline.compute_cov3D_python=true" "++pipeline.covariance_signed_point_json=$POINT_PRIOR_JSON" \\
  "++pipeline.covariance_signed_point_screen_actuator_enable=true" "++pipeline.covariance_signed_shrink_factor=1.0" \\
  "++pipeline.covariance_signed_grow_factor=1.0" "++pipeline.covariance_signed_max_shrink_points=$MAX_POINTS" "++pipeline.covariance_signed_max_grow_points=0" \\
  "++pipeline.covariance_signed_dynamic_enable=false" "++pipeline.covariance_signed_screen_actuator_enable=true" \\
  "++pipeline.covariance_signed_screen_normal_shrink_factor=$SHRINK_FACTOR" "++pipeline.covariance_signed_screen_normal_grow_factor=1.0" "++pipeline.covariance_signed_screen_tangent_factor=1.0" \\
  "++pipeline.covariance_signed_center_offset_enable=true" "++pipeline.covariance_signed_center_offset_outer_px=0.35" "++pipeline.covariance_signed_center_offset_inner_px=0.0"
EOF
  chmod +x "$TRAIN_SCRIPT"; log_event "train_start" "$TRAIN_EXP_DIR"; "$TRAIN_SCRIPT" > "$TRAIN_LAUNCH_LOG" 2>&1 &
  TRAIN_PID="$!"; echo "$TRAIN_PID" > "$LOG_DIR/train.pid"; TRAIN_EST_END_BJT="$(TZ=Asia/Shanghai date -d '+65 minutes' '+%F %T BJT')"; log_event "train_launched" "pid=$TRAIN_PID est_end=$TRAIN_EST_END_BJT"
fi
END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"; log_event "finished_bjt" "$END_BJT"
echo "LOG_DIR=$LOG_DIR"; echo "POINT_PRIOR_JSON=$POINT_PRIOR_JSON"; echo "RAW_GATE_STATUS=$RAW_GATE_STATUS"; echo "RAW_GATE_SUMMARY=$RAW_GATE_SUMMARY"; echo "TRAIN_PID=$TRAIN_PID"; echo "TRAIN_EXP_DIR=$TRAIN_EXP_DIR"; echo "TRAIN_EST_END_BJT=$TRAIN_EST_END_BJT"; echo "END_BJT=$END_BJT"
