#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-stageB_v262_ray_carve_validator_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v262_ray_carve_validator_${RUN_ID}}"
mkdir -p "$LOG_DIR"

NOHUP_LOG="$LOG_DIR/nohup.log"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"

BASE_EXP_DEFAULT="$ROOT/exp/stageB/377_hulk_light_v233d_shoes_preserve_control_stageB_compact_v233_skincloth_20260512_161652_bjt"
BASE_CKPT_DEFAULT="$BASE_EXP_DEFAULT/ckpt135710.pth"

env \
  RUN_ID="$RUN_ID" \
  GPU="$GPU" \
  LOG_DIR="$LOG_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  BASE_EXP="${BASE_EXP:-$BASE_EXP_DEFAULT}" \
  BASE_CKPT="${BASE_CKPT:-$BASE_CKPT_DEFAULT}" \
  WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}" \
  CANDIDATE_VIEWS="${CANDIDATE_VIEWS:-1,2,3,4,5,6,7,8,9,10,11,12}" \
  EVAL_VIEWS="${EVAL_VIEWS:-21,22,23}" \
  FRAMES="${FRAMES:-0,570,60}" \
  MIN_INNER_VIEWS="${MIN_INNER_VIEWS:-2}" \
  MAX_OUTER_VIEWS="${MAX_OUTER_VIEWS:-0}" \
  MIN_HELDOUT_INNER_VIEWS="${MIN_HELDOUT_INNER_VIEWS:-1}" \
  MAX_HELDOUT_OUTER_VIEWS="${MAX_HELDOUT_OUTER_VIEWS:-0}" \
  setsid bash "$ROOT/tools/run_377_stageB_v262_ray_carve_validator_serial.sh" \
  > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "$PID" > "$LOG_DIR/launcher.pid"

{
  echo "RUN_ID=$RUN_ID"
  echo "PID=$PID"
  echo "GPU=$GPU"
  echo "LOG_DIR=$LOG_DIR"
  echo "NOHUP_LOG=$NOHUP_LOG"
  echo "BASE_CKPT=${BASE_CKPT:-$BASE_CKPT_DEFAULT}"
  echo "EVENTS=$LOG_DIR/events.tsv"
  echo "SUMMARY=$LOG_DIR/summary.tsv"
  echo "STATUS_JSON=$LOG_DIR/status.json"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
} | tee "$LOG_DIR/launcher_info.txt"
