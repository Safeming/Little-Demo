#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
RUN_ID="${RUN_ID:-v383_cumulative_canary_selector_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v383_cumulative_canary_selector_${RUN_ID}}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v383_cumulative_canary_selector_${RUN_ID}}"

BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_ASSET_JSON="${BASE_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v374_portfolio_merge_grouped_actuator_v374_v374_v376_queue_20260527_192801_bjt/assets/v374_portfolio_merge_grouped_actuator_asset.json}"
CLOSURE_ASSET_JSON="${CLOSURE_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v382_post_v374_residual_bundle_selector_v382_post_v374_residual_bundle_selector_20260528_182940_bjt/assets/v382_post_v374_residual_bundle_candidate_asset.json}"
PREFILTER_TSV="${PREFILTER_TSV:-$ROOT/exp/stageB/logs/377_explicit_binding_v381_closure_raw_selector_v382_selector_v382_post_v374_residual_bundle_selector_20260528_182940_bjt/action_validation.tsv}"
CANARY_WORST_TSV="${CANARY_WORST_TSV:-$ROOT/exp/formal/logs/377_v338_raw_contour_gate_formal_377_v381_closure_raw_selector_raw_gate_20260528_193641_bjt/worst_frames.tsv}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"

CANARY_TOP_K="${CANARY_TOP_K:-6}"
MAX_CANDIDATES="${MAX_CANDIDATES:-38}"
CHILD_OPACITY="${CHILD_OPACITY:-0.045}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
LAUNCH_LOG="$LOG_DIR/v383_launcher.log"

mkdir -p "$LOG_DIR" "$EXP_ROOT"
START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d '+4 hours 30 minutes' '+%F %T BJT')"
cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_ASSET_JSON=$BASE_ASSET_JSON
CLOSURE_ASSET_JSON=$CLOSURE_ASSET_JSON
PREFILTER_TSV=$PREFILTER_TSV
CANARY_WORST_TSV=$CANARY_WORST_TSV
CANARY_TOP_K=$CANARY_TOP_K
MAX_CANDIDATES=$MAX_CANDIDATES
EOF

env -u LOG_DIR -u EXP_ROOT -u HYDRA_RUN_ROOT \
  GPU="$GPU" \
  PYTHON_BIN="$PYTHON_BIN" \
  CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  RUN_ID="$RUN_ID" \
  BASE_EXP="$BASE_EXP" \
  BASE_ASSET_JSON="$BASE_ASSET_JSON" \
  CLOSURE_ASSET_JSON="$CLOSURE_ASSET_JSON" \
  PREFILTER_TSV="$PREFILTER_TSV" \
  CANARY_WORST_TSV="$CANARY_WORST_TSV" \
  CANARY_TOP_K="$CANARY_TOP_K" \
  MAX_CANDIDATES="$MAX_CANDIDATES" \
  CANDIDATE_CKPT="$CANDIDATE_CKPT" \
  CHILD_OPACITY="$CHILD_OPACITY" \
  TRAIN_STEPS="$TRAIN_STEPS" \
  "$PYTHON_BIN" tools/run_377_explicit_binding_v383_cumulative_canary_selector.py \
  --log-dir "$LOG_DIR" \
  --exp-root "$EXP_ROOT" \
  > "$LAUNCH_LOG" 2>&1

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "LOG_DIR=$LOG_DIR"
echo "EXP_ROOT=$EXP_ROOT"
echo "LAUNCH_LOG=$LAUNCH_LOG"
echo "START_BJT=$START_BJT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
